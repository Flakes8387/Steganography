"""
Deep Learning Training Script for Steganography

Complete training loop with:
- Dataset loading
- Encoder → Distortions → Decoder pipeline
- Loss computation (image + message)
- Backpropagation and optimization
- Checkpoint saving
- TensorBoard logging
"""

from defense import DenoiseBeforeDecode, AdaptiveAntiAlias
from utils.config_loader import load_config, merge_config_with_args, print_config
from attacks import JPEGCompression, GaussianNoise, ResizeAttack, ColorJitter
from models.model import StegoModel
import matplotlib.pyplot as plt
import os
import sys
import argparse
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
import glob
from torch.cuda.amp import autocast, GradScaler
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML not installed. Install it with: pip install pyyaml")
    sys.exit(1)


# =============================================================================
# Dataset Class
# =============================================================================
class SteganographyDataset(Dataset):
    """
    Dataset for loading images and generating random binary messages.

    Supports:
    - Patch-based loading for high-resolution images (e.g., DIV2K)
    - Random/center cropping to target size
    - Automatic normalization to [0,1]
    """

    def __init__(self, image_dir, message_length=1024, image_size=256, max_images=1000,
                 use_patches=True, patches_per_image=4, random_crop=True):
        """
        Args:
            image_dir: Directory containing images
            message_length: Length of binary messages
            image_size: Size to crop/resize images to (e.g., 128 for DIV2K patches)
            max_images: Maximum number of images to load (default: 1000 for local GPU training)
            use_patches: If True, extract multiple patches per image (increases dataset size)
            patches_per_image: Number of patches to extract per image (only if use_patches=True)
            random_crop: If True, use random crops; if False, use center crop
        """
        self.image_dir = image_dir
        self.message_length = message_length
        self.image_size = image_size
        self.use_patches = use_patches
        self.patches_per_image = patches_per_image
        self.random_crop = random_crop

        # Find all images (prioritize .png for DIV2K, but support all formats)
        self.image_paths = []
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
            self.image_paths.extend(glob.glob(os.path.join(
                image_dir, '**', f'*{ext}'), recursive=True))

        # Shuffle all images first for random sampling
        import random
        random.shuffle(self.image_paths)

        # Limit dataset size for reasonable local GPU training time (500-1000 images)
        # This limits BASE images, not total patches
        if max_images is not None and max_images < len(self.image_paths):
            self.image_paths = self.image_paths[:max_images]
            print(
                f"Loaded {len(self.image_paths)} images from {image_dir} (limited to {max_images} for local GPU)")
        else:
            print(f"Loaded {len(self.image_paths)} images from {image_dir}")

        # Calculate total dataset size with patches
        if use_patches:
            self.total_samples = len(self.image_paths) * patches_per_image
            print(f"[OK] Patch-based loading enabled:")
            print(f"  - Patches per image: {patches_per_image}")
            print(f"  - Base images: {len(self.image_paths)}")
            print(f"  - Total training samples: {self.total_samples}")
            print(f"  - Patches are randomly sampled each epoch for diversity")
        else:
            self.total_samples = len(self.image_paths)

        # Setup transforms
        if use_patches:
            # For patch-based loading (DIV2K): random/center crop to target size
            if random_crop:
                self.transform = transforms.Compose([
                    transforms.RandomCrop(image_size),  # Extract random patch
                    transforms.ToTensor(),  # Converts to [0, 1] and CHW format
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.CenterCrop(image_size),  # Extract center patch
                    transforms.ToTensor(),
                ])
        else:
            # Standard resize for smaller images
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),  # Converts to [0, 1]
            ])

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # Map sample index to image index
        if self.use_patches:
            image_idx = idx // self.patches_per_image
            patch_idx = idx % self.patches_per_image
        else:
            image_idx = idx
            patch_idx = 0

        # Load image
        try:
            image_path = self.image_paths[image_idx]
            image = Image.open(image_path).convert('RGB')

            # For patch-based loading, ensure image is large enough
            if self.use_patches:
                width, height = image.size
                if width < self.image_size or height < self.image_size:
                    # If image is smaller than patch size, resize it first
                    scale = max(self.image_size / width,
                                self.image_size / height)
                    new_width = int(width * scale) + 1
                    new_height = int(height * scale) + 1
                    image = image.resize(
                        (new_width, new_height), Image.BICUBIC)

            # Apply transform (crop and normalize to [0,1])
            image = self.transform(image)

        except Exception as e:
            print(f"Error loading {self.image_paths[image_idx]}: {e}")
            # Return next image
            return self.__getitem__((idx + 1) % len(self))

        # Generate random binary message
        message = torch.randint(0, 2, (self.message_length,)).float()

        return image, message


# =============================================================================
# Training Function
# =============================================================================
def train_epoch(model, dataloader, optimizer, device, epoch, writer, enable_distortions=False, apply_attacks=False, use_amp=True, jpeg_only=False):
    """
    Train for one epoch.

    Args:
        model: StegoModel instance
        dataloader: DataLoader for training data
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        writer: TensorBoard writer
        enable_distortions: Whether to enable built-in distortions (JPEG, noise, etc.)
        apply_attacks: Whether to apply additional random attacks during training
        use_amp: Use automatic mixed precision for faster training (default: True)
        jpeg_only: If True, only apply JPEG compression (for compute-limited local training)

    Returns:
        Dictionary of average metrics
    """
    model.train()

    # Initialize GradScaler for mixed precision training
    scaler = GradScaler(enabled=use_amp and device.type == 'cuda')

    # Initialize attack modules (optional)
    if apply_attacks:
        jpeg = JPEGCompression(quality_range=(50, 95)).to(device)
        noise = GaussianNoise(std_range=(0.01, 0.03)).to(device)
        resize = ResizeAttack(scale_range=(0.7, 0.9)).to(device)
        color_jitter = ColorJitter().to(device)

    total_loss = 0.0
    total_image_loss = 0.0
    total_message_loss = 0.0
    total_ber = 0.0
    total_bit_accuracy = 0.0
    total_pixel_delta = 0.0

    start_time = time.time()

    for batch_idx, (cover_images, binary_messages) in enumerate(dataloader):
        # Move to device (non_blocking for async transfer with pinned memory)
        cover_images = cover_images.to(device, non_blocking=True)
        binary_messages = binary_messages.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Forward pass with mixed precision
        with autocast(enabled=use_amp and device.type == 'cuda'):
            # Forward pass: encoder → distortions → decoder
            # Apply distortions only if enabled (default: clean training until 75% accuracy)
            outputs = model.forward(
                cover_images, binary_messages, apply_distortions=enable_distortions, jpeg_only=jpeg_only)

            stego_images = outputs['stego_image']
            distorted_stego = outputs['distorted_stego']
            decoded_logits = outputs['decoded_logits']
            decoded_message = outputs['decoded_message']

            # Optionally apply additional attacks
            if apply_attacks and torch.rand(1).item() < 0.3:
                attack_type = torch.rand(1).item()
                if attack_type < 0.25:
                    distorted_stego = jpeg(distorted_stego)
                elif attack_type < 0.5:
                    distorted_stego = noise(distorted_stego)
                elif attack_type < 0.75:
                    distorted_stego = resize(distorted_stego)
                else:
                    distorted_stego = color_jitter(distorted_stego)

                # Re-decode after additional attack
                decoded_logits = model.decode(
                    distorted_stego, return_logits=True)
                decoded_message = (torch.sigmoid(decoded_logits) > 0.5).float()

            # Compute losses
            image_loss = nn.functional.mse_loss(stego_images, cover_images)
            message_loss = nn.functional.binary_cross_entropy_with_logits(
                decoded_logits, binary_messages)

            # Combined loss with weighting (message loss weighted 5x more)
            total_loss_batch = image_loss + 5.0 * message_loss

        # Backward pass with gradient scaling (for mixed precision)
        scaler.scale(total_loss_batch).backward()

        # Gradient clipping (optional, prevents exploding gradients)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        # Calculate BER (Bit Error Rate) and bit accuracy
        BER = (decoded_message != binary_messages).float().mean()
        bit_accuracy = 1.0 - BER

        # Calculate pixel delta (debug metric for imperceptibility)
        pixel_delta = torch.mean(torch.abs(stego_images - cover_images))

        # Accumulate metrics
        total_loss += total_loss_batch.item()
        total_image_loss += image_loss.item()
        total_message_loss += message_loss.item()
        total_ber += BER.item()
        total_bit_accuracy += bit_accuracy.item()
        total_pixel_delta += pixel_delta.item()

        # Log to TensorBoard (every 10 batches)
        if batch_idx % 10 == 0:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar(
                'Train/Loss', total_loss_batch.item(), global_step)
            writer.add_scalar('Train/ImageLoss',
                              image_loss.item(), global_step)
            writer.add_scalar('Train/MessageLoss',
                              message_loss.item(), global_step)
            writer.add_scalar('Train/BER', BER.item(), global_step)
            writer.add_scalar('Train/BitAccuracy',
                              bit_accuracy.item(), global_step)
            writer.add_scalar('Train/PixelDelta',
                              pixel_delta.item(), global_step)

            # Check pixel delta range (should be 0.005-0.02 for good imperceptibility)
            pixel_delta_val = pixel_delta.item()
            delta_status = ""
            if pixel_delta_val < 0.005:
                delta_status = " [WARNING] TOO LOW"
            elif pixel_delta_val > 0.02:
                delta_status = " [WARNING] TOO HIGH"
            else:
                delta_status = " [OK]"

            # Print progress (compact format for DIV2K)
            batch_pct = (batch_idx + 1) / len(dataloader) * 100
            print(f"[Epoch {epoch+1:3d}] [{batch_pct:5.1f}%] "
                  f"BitAcc: {bit_accuracy.item()*100:5.2f}% | "
                  f"BER: {BER.item():.4f} | "
                  f"Loss: {total_loss_batch.item():.4f} | "
                  f"PixelΔ: {pixel_delta_val:.4f}{delta_status}")

    # Calculate averages
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_image_loss = total_image_loss / num_batches
    avg_message_loss = total_message_loss / num_batches
    avg_ber = total_ber / num_batches
    avg_bit_accuracy = total_bit_accuracy / num_batches
    avg_pixel_delta = total_pixel_delta / num_batches

    epoch_time = time.time() - start_time

    # Log epoch-level metrics to TensorBoard
    writer.add_scalar('Train/EpochLoss', avg_loss, epoch)
    writer.add_scalar('Train/EpochImageLoss', avg_image_loss, epoch)
    writer.add_scalar('Train/EpochMessageLoss', avg_message_loss, epoch)
    writer.add_scalar('Train/EpochBER', avg_ber, epoch)
    writer.add_scalar('Train/EpochBitAccuracy', avg_bit_accuracy, epoch)
    writer.add_scalar('Train/EpochPixelDelta', avg_pixel_delta, epoch)

    return {
        'loss': avg_loss,
        'image_loss': avg_image_loss,
        'message_loss': avg_message_loss,
        'ber': avg_ber,
        'bit_accuracy': avg_bit_accuracy,
        'pixel_delta': avg_pixel_delta,
        'time': epoch_time
    }


# =============================================================================
# Validation Function
# =============================================================================
def validate(model, dataloader, device, epoch, writer):
    """
    Validate the model.

    Args:
        model: StegoModel instance
        dataloader: DataLoader for validation data
        device: Device to validate on
        epoch: Current epoch number
        writer: TensorBoard writer

    Returns:
        Dictionary of average metrics
    """
    model.eval()

    total_loss = 0.0
    total_image_loss = 0.0
    total_message_loss = 0.0
    total_ber = 0.0
    total_bit_accuracy = 0.0

    with torch.no_grad():
        for batch_idx, (cover_images, binary_messages) in enumerate(dataloader):
            # Move to device
            cover_images = cover_images.to(device)
            binary_messages = binary_messages.to(device)

            # Forward pass without distortions for clean validation
            outputs = model.forward(
                cover_images, binary_messages, apply_distortions=False)

            stego_images = outputs['stego_image']
            decoded_logits = outputs['decoded_logits']
            decoded_message = outputs['decoded_message']

            # Compute losses
            image_loss = nn.functional.mse_loss(stego_images, cover_images)
            message_loss = nn.functional.binary_cross_entropy_with_logits(
                decoded_logits, binary_messages)
            # Combined loss with weighting (message loss weighted 5x more)
            total_loss_batch = image_loss + 5.0 * message_loss

            # Calculate BER (Bit Error Rate) and bit accuracy
            BER = (decoded_message != binary_messages).float().mean()
            bit_accuracy = 1.0 - BER

            # Accumulate metrics
            total_loss += total_loss_batch.item()
            total_image_loss += image_loss.item()
            total_message_loss += message_loss.item()
            total_ber += BER.item()
            total_bit_accuracy += bit_accuracy.item()

    # Calculate averages
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_image_loss = total_image_loss / num_batches
    avg_message_loss = total_message_loss / num_batches
    avg_ber = total_ber / num_batches
    avg_bit_accuracy = total_bit_accuracy / num_batches

    # Log to TensorBoard
    writer.add_scalar('Val/Loss', avg_loss, epoch)
    writer.add_scalar('Val/ImageLoss', avg_image_loss, epoch)
    writer.add_scalar('Val/MessageLoss', avg_message_loss, epoch)
    writer.add_scalar('Val/BER', avg_ber, epoch)
    writer.add_scalar('Val/BitAccuracy', avg_bit_accuracy, epoch)

    return {
        'loss': avg_loss,
        'image_loss': avg_image_loss,
        'message_loss': avg_message_loss,
        'ber': avg_ber,
        'bit_accuracy': avg_bit_accuracy
    }


# =============================================================================
# Checkpoint Saving/Loading
# =============================================================================
def save_training_plots(metrics_history, save_dir):
    """
    Generate and save training plots for analysis and reports.

    Args:
        metrics_history: Dictionary containing lists of metrics per epoch
        save_dir: Directory to save plots
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    epochs = list(range(1, len(metrics_history['train_bit_accuracy']) + 1))

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Metrics Over Time', fontsize=16, fontweight='bold')

    # Plot 1: Bit Accuracy vs Epochs
    ax1 = axes[0, 0]
    ax1.plot(
        epochs, metrics_history['train_bit_accuracy'], 'b-', label='Train', linewidth=2)
    if 'val_bit_accuracy' in metrics_history and metrics_history['val_bit_accuracy']:
        ax1.plot(epochs, metrics_history['val_bit_accuracy'],
                 'r--', label='Validation', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Bit Accuracy', fontsize=12)
    ax1.set_title('Bit Accuracy vs Epochs', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])

    # Plot 2: BER vs Epochs
    ax2 = axes[0, 1]
    ax2.plot(epochs, metrics_history['train_ber'],
             'b-', label='Train', linewidth=2)
    if 'val_ber' in metrics_history and metrics_history['val_ber']:
        ax2.plot(epochs, metrics_history['val_ber'],
                 'r--', label='Validation', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Bit Error Rate (BER)', fontsize=12)
    ax2.set_title('BER vs Epochs', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 0.5])

    # Plot 3: Pixel Distortion vs Epochs (if available)
    ax3 = axes[1, 0]
    if 'pixel_delta' in metrics_history and metrics_history['pixel_delta']:
        ax3.plot(epochs, metrics_history['pixel_delta'], 'g-', linewidth=2)
        ax3.axhline(y=0.005, color='orange', linestyle='--',
                    label='Min threshold (0.005)', alpha=0.7)
        ax3.axhline(y=0.02, color='red', linestyle='--',
                    label='Max threshold (0.02)', alpha=0.7)
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('Mean Pixel Delta', fontsize=12)
        ax3.set_title('Pixel Distortion vs Epochs',
                      fontsize=14, fontweight='bold')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Pixel Delta\nNot Available',
                 ha='center', va='center', fontsize=12, transform=ax3.transAxes)
        ax3.set_title('Pixel Distortion vs Epochs',
                      fontsize=14, fontweight='bold')

    # Plot 4: Loss vs Epochs
    ax4 = axes[1, 1]
    ax4.plot(epochs, metrics_history['train_loss'],
             'b-', label='Train Total Loss', linewidth=2)
    if 'val_loss' in metrics_history and metrics_history['val_loss']:
        ax4.plot(epochs, metrics_history['val_loss'],
                 'r--', label='Validation Loss', linewidth=2)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Loss', fontsize=12)
    ax4.set_title('Loss vs Epochs', fontsize=14, fontweight='bold')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plots
    plot_path = save_dir / 'training_metrics.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Training plots saved to {plot_path}")

    # Also save individual plots for reports
    # Bit Accuracy
    fig_acc, ax_acc = plt.subplots(figsize=(8, 6))
    ax_acc.plot(epochs, metrics_history['train_bit_accuracy'],
                'b-', label='Train', linewidth=2, marker='o', markersize=4)
    if 'val_bit_accuracy' in metrics_history and metrics_history['val_bit_accuracy']:
        ax_acc.plot(epochs, metrics_history['val_bit_accuracy'], 'r--',
                    label='Validation', linewidth=2, marker='s', markersize=4)
    ax_acc.set_xlabel('Epoch', fontsize=14)
    ax_acc.set_ylabel('Bit Accuracy', fontsize=14)
    ax_acc.set_title('Bit Accuracy vs Epochs', fontsize=16, fontweight='bold')
    ax_acc.legend(loc='lower right', fontsize=12)
    ax_acc.grid(True, alpha=0.3)
    ax_acc.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(save_dir / 'bit_accuracy_plot.png',
                dpi=300, bbox_inches='tight')
    plt.close(fig_acc)

    # BER
    fig_ber, ax_ber = plt.subplots(figsize=(8, 6))
    ax_ber.plot(epochs, metrics_history['train_ber'], 'b-',
                label='Train', linewidth=2, marker='o', markersize=4)
    if 'val_ber' in metrics_history and metrics_history['val_ber']:
        ax_ber.plot(epochs, metrics_history['val_ber'], 'r--',
                    label='Validation', linewidth=2, marker='s', markersize=4)
    ax_ber.set_xlabel('Epoch', fontsize=14)
    ax_ber.set_ylabel('Bit Error Rate (BER)', fontsize=14)
    ax_ber.set_title('BER vs Epochs', fontsize=16, fontweight='bold')
    ax_ber.legend(loc='upper right', fontsize=12)
    ax_ber.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'ber_plot.png', dpi=300, bbox_inches='tight')
    plt.close(fig_ber)

    # Pixel Delta (if available)
    if 'pixel_delta' in metrics_history and metrics_history['pixel_delta']:
        fig_delta, ax_delta = plt.subplots(figsize=(8, 6))
        ax_delta.plot(
            epochs, metrics_history['pixel_delta'], 'g-', linewidth=2, marker='o', markersize=4)
        ax_delta.axhline(y=0.005, color='orange', linestyle='--',
                         label='Min threshold (0.005)', alpha=0.7, linewidth=2)
        ax_delta.axhline(y=0.02, color='red', linestyle='--',
                         label='Max threshold (0.02)', alpha=0.7, linewidth=2)
        ax_delta.fill_between(epochs, 0.005, 0.02, alpha=0.1,
                              color='green', label='Optimal range')
        ax_delta.set_xlabel('Epoch', fontsize=14)
        ax_delta.set_ylabel('Mean Pixel Delta', fontsize=14)
        ax_delta.set_title('Pixel Distortion vs Epochs',
                           fontsize=16, fontweight='bold')
        ax_delta.legend(loc='upper right', fontsize=12)
        ax_delta.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / 'pixel_delta_plot.png',
                    dpi=300, bbox_inches='tight')
        plt.close(fig_delta)

    plt.close(fig)
    print(f"📈 Individual plots saved: bit_accuracy_plot.png, ber_plot.png, pixel_delta_plot.png")


def save_checkpoint(model, optimizer, epoch, metrics, checkpoint_dir, is_best=False):
    """
    Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state to save
        epoch: Current epoch
        metrics: Dictionary of metrics
        checkpoint_dir: Directory to save checkpoint
        is_best: Whether this is the best model so far
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }

    # Save latest checkpoint
    checkpoint_path = checkpoint_dir / 'checkpoint_latest.pth'
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

    # Save best checkpoint
    if is_best:
        best_path = checkpoint_dir / 'best_model_local.pth'
        torch.save(checkpoint, best_path)
        print(f"[OK] Saved best model to {best_path}")

    # Save periodic checkpoint
    if epoch % 10 == 0:
        periodic_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, periodic_path)


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """
    Load model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)

    Returns:
        Dictionary containing epoch and metrics
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"Resuming from epoch {checkpoint['epoch']}")

    return {
        'epoch': checkpoint['epoch'],
        'metrics': checkpoint.get('metrics', {})
    }


# =============================================================================
# Main Training Loop
# =============================================================================
def main(args):
    """Main training function."""

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Setup device - Force CUDA/GPU usage
    print("=" * 60)
    print("GPU DETECTION AND SETUP")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("[ERROR] CUDA is NOT available!")
        print("[WARNING]  This training requires a CUDA-capable NVIDIA GPU.")
        print("\nPossible solutions:")
        print("1. Install CUDA-enabled PyTorch:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("2. Check if your GPU is CUDA-compatible")
        print("3. Update NVIDIA drivers")
        print("\n[ERROR] ABORTING TRAINING - GPU is required!")
        print("=" * 60)
        sys.exit(1)

    # Use CUDA device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Print GPU information
    print(f"[SUCCESS] CUDA is available!")
    print(f"   GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   Number of GPUs: {torch.cuda.device_count()}")

    # Get GPU memory info
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"   Total GPU Memory: {gpu_memory:.2f} GB")

    print(f"\n[SUCCESS] Using device: {device}")
    print("=" * 60)
    print()

    # Create directories
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(args.log_dir) / datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir.mkdir(parents=True, exist_ok=True)

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs saved to {log_dir}")
    print(f"Run: tensorboard --logdir {args.log_dir}")

    # Create datasets
    print("\n" + "=" * 60)
    print("DATASET PREPARATION")
    print("=" * 60)

    # Print dataset configuration
    print(f"\nDataset Configuration:")
    print(f"  Type: {args.dataset_type.upper()}")
    print(f"  Path: {args.train_dir}")
    print(f"  Image size (patch size): {args.image_size}x{args.image_size}")
    print(f"  Message length: {args.message_length} bits")

    if args.use_patches:
        print(f"  Patch-based loading: ENABLED")
        print(f"  Patches per image: {args.patches_per_image}")
        print(f"  Random crop: {args.random_crop}")
        if args.max_train_images:
            total_samples = args.max_train_images * args.patches_per_image
            print(f"  Max base images: {args.max_train_images}")
            print(
                f"  Total training samples: {total_samples} ({args.max_train_images} x {args.patches_per_image})")
    else:
        print(f"  Patch-based loading: DISABLED")
        if args.max_train_images:
            print(f"  Max training images: {args.max_train_images}")

    print(f"\nLoading training images from: {args.train_dir}")
    train_dataset = SteganographyDataset(
        image_dir=args.train_dir,
        message_length=args.message_length,
        image_size=args.image_size,
        max_images=args.max_train_images,
        use_patches=args.use_patches,
        patches_per_image=args.patches_per_image,
        random_crop=args.random_crop
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,  # Shuffle images each epoch for better convergence
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    # Print dataset summary
    print(f"\n[SUCCESS] Dataset loaded successfully:")
    if args.use_patches:
        num_base_images = len(train_dataset.image_paths)
        num_total_samples = len(train_dataset)
        print(f"   Base images: {num_base_images}")
        print(f"   Patches per image: {args.patches_per_image}")
        print(f"   Total training samples: {num_total_samples}")
        print(f"   Patch size: {args.image_size}x{args.image_size}")
    else:
        print(f"   Training images: {len(train_dataset)}")
        print(f"   Image size: {args.image_size}x{args.image_size}")
    print(f"   Message length: {args.message_length} bits")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Training batches per epoch: {len(train_loader)}")

    # Dataset Verification Step - Load and inspect one batch
    print("\n" + "=" * 60)
    print("DATASET VERIFICATION")
    print("=" * 60)
    print("\nLoading one sample batch to verify dataset integrity...\n")

    try:
        # Get one batch
        sample_images, sample_messages = next(iter(train_loader))

        # Print tensor shapes
        print(f"Batch Shapes:")
        print(
            f"  Images:   {list(sample_images.shape)} (batch, channels, height, width)")
        print(
            f"  Messages: {list(sample_messages.shape)} (batch, message_length)")

        # Verify tensor properties
        print(f"\nTensor Properties:")
        print(f"  Image dtype:   {sample_images.dtype}")
        print(
            f"  Image range:   [{sample_images.min():.3f}, {sample_images.max():.3f}]")
        print(f"  Message dtype: {sample_messages.dtype}")
        print(
            f"  Message range: [{sample_messages.min():.0f}, {sample_messages.max():.0f}]")

        # Display first sample
        print(f"\nFirst Sample in Batch:")
        first_image = sample_images[0]
        first_message = sample_messages[0]

        print(f"  Image shape: {list(first_image.shape)}")
        print(
            f"  Image stats: mean={first_image.mean():.3f}, std={first_image.std():.3f}")

        # Print first 32 bits of message (or all if shorter)
        msg_preview = first_message[:min(32, len(first_message))].cpu().numpy()
        msg_str = ''.join(str(int(bit)) for bit in msg_preview)
        if len(first_message) > 32:
            msg_str += f"... ({len(first_message)} bits total)"
        print(f"  Message bits: {msg_str}")

        # Save first sample image for inspection
        try:
            import torchvision
            sample_image_path = os.path.join(
                args.checkpoint_dir, 'dataset_sample.png')
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            torchvision.utils.save_image(first_image, sample_image_path)
            print(f"  Saved sample image: {sample_image_path}")
        except Exception as e:
            print(f"  (Could not save sample image: {e})")

        # Validation checks
        print(f"\nValidation Checks:")
        checks_passed = True

        # Check 1: Image values in [0, 1]
        if sample_images.min() >= 0.0 and sample_images.max() <= 1.0:
            print(f"  [OK] Images normalized to [0, 1]")
        else:
            print(f"  [WARNING] Images not in [0, 1] range")
            checks_passed = False

        # Check 2: Message values are binary
        unique_values = torch.unique(sample_messages)
        if len(unique_values) <= 2 and all(v in [0, 1] for v in unique_values):
            print(f"  [OK] Messages are binary (0/1)")
        else:
            print(
                f"  [WARNING] Messages contain non-binary values: {unique_values.tolist()}")
            checks_passed = False

        # Check 3: Batch size matches
        if sample_images.shape[0] == args.batch_size:
            print(f"  [OK] Batch size correct: {sample_images.shape[0]}")
        else:
            print(
                f"  [INFO] Batch size: {sample_images.shape[0]} (expected: {args.batch_size}, last batch may differ)")

        # Check 4: Image dimensions match
        if sample_images.shape[2] == args.image_size and sample_images.shape[3] == args.image_size:
            print(
                f"  [OK] Image dimensions correct: {args.image_size}x{args.image_size}")
        else:
            print(
                f"  [ERROR] Image dimensions {sample_images.shape[2]}x{sample_images.shape[3]} != {args.image_size}x{args.image_size}")
            checks_passed = False

        # Check 5: Message length matches
        if sample_messages.shape[1] == args.message_length:
            print(f"  [OK] Message length correct: {args.message_length} bits")
        else:
            print(
                f"  [ERROR] Message length {sample_messages.shape[1]} != {args.message_length}")
            checks_passed = False

        if checks_passed:
            print(f"\n[SUCCESS] All dataset verification checks passed!")
        else:
            print(
                f"\n[WARNING] Some verification checks failed - review warnings above")
            if not checks_passed:
                response = input("\nContinue training anyway? (y/n): ")
                if response.lower() != 'y':
                    print("Training aborted by user.")
                    sys.exit(1)

        print("=" * 60)

    except Exception as e:
        print(f"\n[ERROR] ERROR during dataset verification: {e}")
        print(f"   This indicates a problem with the dataset or data loader.")
        import traceback
        traceback.print_exc()
        print("\nAborting training due to dataset verification failure.")
        sys.exit(1)

    # Validation dataset (optional)
    val_loader = None
    if args.val_dir and os.path.exists(args.val_dir):
        val_dataset = SteganographyDataset(
            image_dir=args.val_dir,
            message_length=args.message_length,
            image_size=args.image_size,
            max_images=args.max_val_images,
            # Disable patches for validation (consistent evaluation)
            use_patches=False,
            random_crop=False  # Use center crop for validation
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available()
        )
        print(f"Validation dataset: {len(val_dataset)} images")

    # Initialize model
    print("\n" + "=" * 60)
    print("MODEL INITIALIZATION")
    print("=" * 60)
    print(f"\nInitializing StegoModel...")
    model = StegoModel(
        message_length=args.message_length,
        image_size=args.image_size,
        enable_distortions=args.enable_distortions
    )
    model = model.to(device)

    # Enable cuDNN optimizations for faster training
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print("[OK] cuDNN benchmark enabled for faster convolutions")

        # Check if mixed precision is supported
        amp_supported = torch.cuda.get_device_capability()[
            0] >= 7  # Volta or newer
        if amp_supported:
            print(
                "[OK] Mixed precision (AMP) enabled - faster training with lower memory")
        else:
            print(
                "[WARNING]  Mixed precision not supported on this GPU (requires compute capability >= 7.0)")

    # Print model info
    params = model.get_num_parameters()
    print(f"Encoder parameters: {params['encoder']:,}")
    print(f"Decoder parameters: {params['decoder']:,}")
    print(f"Total parameters: {params['total']:,}")

    # Initialize optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    best_bit_accuracy = 0.0
    epochs_without_improvement = 0
    early_stop_patience = 25

    # Initialize metrics history for plotting
    metrics_history = {
        'train_bit_accuracy': [],
        'train_ber': [],
        'train_loss': [],
        'val_bit_accuracy': [],
        'val_ber': [],
        'val_loss': [],
        'pixel_delta': []
    }

    if args.resume:
        if os.path.exists(args.resume):
            checkpoint_info = load_checkpoint(args.resume, model, optimizer)
            start_epoch = checkpoint_info['epoch'] + 1
            print(f"Resuming training from epoch {start_epoch}")
        else:
            print(f"Checkpoint {args.resume} not found, starting from scratch")

    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    # Track if distortions have been enabled
    distortions_enabled = args.enable_distortions
    jpeg_only_mode = False  # Start with JPEG-only mode when threshold reached
    distortion_threshold = 0.75  # Enable distortions when accuracy >= 75%

    if not distortions_enabled:
        print("\n[WARNING]  DISTORTIONS DISABLED: Clean-image training mode")
        print(
            f"   JPEG compression (0.1 probability) will auto-enable when bit accuracy >= {distortion_threshold*100:.0f}%")
        print(
            f"   Other distortions (resize, noise) kept disabled for local GPU efficiency")
        print("=" * 60)

    for epoch in range(start_epoch, args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        print("-" * 60)

        # Train with mixed precision if GPU supports it
        use_amp = torch.cuda.is_available(
        ) and torch.cuda.get_device_capability()[0] >= 7
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, epoch, writer,
            enable_distortions=distortions_enabled,
            apply_attacks=args.apply_attacks,
            use_amp=use_amp,
            jpeg_only=jpeg_only_mode
        )

        # Enhanced metrics display for DIV2K training
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch + 1}/{args.num_epochs} SUMMARY")
        print(f"{'='*60}")

        # Primary metrics (most important for steganography)
        bit_acc_pct = train_metrics['bit_accuracy'] * 100
        ber_value = train_metrics['ber']

        # Color-coded bit accuracy status
        if bit_acc_pct >= 95:
            acc_status = "[EXCELLENT]"
        elif bit_acc_pct >= 85:
            acc_status = "[GOOD]"
        elif bit_acc_pct >= 75:
            acc_status = "[OK]"
        else:
            acc_status = "[TRAINING]"

        print(f"\nMessage Recovery Performance:")
        print(f"  Bit Accuracy : {bit_acc_pct:6.2f}% {acc_status}")
        print(f"  BER          : {ber_value:.6f}")

        # Loss metrics
        print(f"\nLoss Breakdown:")
        print(f"  Total Loss   : {train_metrics['loss']:.6f}")
        print(f"  Image Loss   : {train_metrics['image_loss']:.6f}")
        print(f"  Message Loss : {train_metrics['message_loss']:.6f}")

        # Image quality metric
        if 'pixel_delta' in train_metrics:
            pixel_delta = train_metrics['pixel_delta']
            if pixel_delta < 0.005:
                delta_status = "[WARNING: Too imperceptible]"
            elif pixel_delta > 0.02:
                delta_status = "[WARNING: Too visible]"
            else:
                delta_status = "[OK]"
            print(f"\nImage Quality:")
            print(f"  Pixel Delta  : {pixel_delta:.6f} {delta_status}")

        # Training speed
        epoch_time = train_metrics['time']
        samples_per_sec = len(train_loader.dataset) / epoch_time
        print(f"\nTraining Speed:")
        print(f"  Epoch Time   : {epoch_time:.1f}s ({epoch_time/60:.1f} min)")
        print(f"  Throughput   : {samples_per_sec:.1f} samples/sec")

        # Track metrics for plotting
        metrics_history['train_bit_accuracy'].append(
            train_metrics['bit_accuracy'])
        metrics_history['train_ber'].append(train_metrics['ber'])
        metrics_history['train_loss'].append(train_metrics['loss'])
        if 'pixel_delta' in train_metrics:
            metrics_history['pixel_delta'].append(train_metrics['pixel_delta'])

        # Auto-enable JPEG-only distortions when bit accuracy threshold is reached
        if not distortions_enabled and train_metrics['bit_accuracy'] >= distortion_threshold:
            distortions_enabled = True
            jpeg_only_mode = True
            print("\n" + "=" * 60)
            print("🎯 JPEG COMPRESSION AUTO-ENABLED!")
            print(
                f"   Bit accuracy reached {train_metrics['bit_accuracy']*100:.2f}% (>= {distortion_threshold*100:.0f}%)")
            print(f"   BER: {train_metrics['ber']:.6f}")
            print("   Switching to Phase 2: JPEG compression with 0.1 probability")
            print("   Other distortions (resize, noise) remain disabled for local GPU")
            print("=" * 60)

        # Validate
        if val_loader is not None:
            val_metrics = validate(model, val_loader, device, epoch, writer)

            print(f"\nValidation Results:")
            print(f"  Bit Accuracy : {val_metrics['bit_accuracy']*100:6.2f}%")
            print(f"  BER          : {val_metrics['ber']:.6f}")
            print(f"  Total Loss   : {val_metrics['loss']:.6f}")

            # Track validation metrics for plotting
            metrics_history['val_bit_accuracy'].append(
                val_metrics['bit_accuracy'])
            metrics_history['val_ber'].append(val_metrics['ber'])
            metrics_history['val_loss'].append(val_metrics['loss'])

            # Update learning rate based on validation loss
            scheduler.step(val_metrics['loss'])

            # Check if best model based on bit accuracy
            is_best = val_metrics['bit_accuracy'] > best_bit_accuracy
            if is_best:
                best_bit_accuracy = val_metrics['bit_accuracy']
                epochs_without_improvement = 0
                improvement = val_metrics['bit_accuracy'] - best_bit_accuracy
                print(f"\n[SUCCESS] NEW BEST MODEL!")
                print(f"  Previous Best : {best_bit_accuracy*100:.2f}%")
                print(
                    f"  Current       : {val_metrics['bit_accuracy']*100:.2f}%")
                print(f"  Improvement   : +{improvement*100:.2f}%")
            else:
                epochs_without_improvement += 1
                print(
                    f"\n  No improvement: {epochs_without_improvement}/{early_stop_patience} epochs")
                print(f"  Best so far   : {best_bit_accuracy*100:.2f}%")
        else:
            # No validation set - use training bit accuracy
            is_best = train_metrics['bit_accuracy'] > best_bit_accuracy
            if is_best:
                improvement = train_metrics['bit_accuracy'] - best_bit_accuracy
                best_bit_accuracy = train_metrics['bit_accuracy']
                epochs_without_improvement = 0
                print(f"\n[SUCCESS] NEW BEST MODEL!")
                print(
                    f"  Previous Best : {(best_bit_accuracy-improvement)*100:.2f}%")
                print(f"  Current       : {best_bit_accuracy*100:.2f}%")
                print(f"  Improvement   : +{improvement*100:.2f}%")
            else:
                epochs_without_improvement += 1
                print(
                    f"\n  No improvement: {epochs_without_improvement}/{early_stop_patience} epochs")
                print(f"  Best so far   : {best_bit_accuracy*100:.2f}%")

        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0 or is_best:
            save_checkpoint(
                model, optimizer, epoch, train_metrics,
                checkpoint_dir, is_best=is_best
            )

            # Log checkpoint save info
            if is_best:
                print(f"  Saved: checkpoints/best_model_local.pth")
            if (epoch + 1) % args.save_freq == 0:
                print(f"  Saved: checkpoints/checkpoint_latest.pth")

        # Log learning rate
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Train/LearningRate', current_lr, epoch)

        # End-of-epoch status line
        print(f"{'='*60}")
        print(f"Learning Rate: {current_lr:.6f}")
        print(f"Best Model   : {best_bit_accuracy*100:.2f}% bit accuracy")
        print(f"{'='*60}\n")

        # Early stopping check
        if epochs_without_improvement >= early_stop_patience:
            print("\n" + "=" * 60)
            print("[STOP] EARLY STOPPING TRIGGERED")
            print(
                f"   No improvement in bit accuracy for {early_stop_patience} epochs")
            print(f"   Best bit accuracy: {best_bit_accuracy*100:.2f}%")
            print(f"   Stopping training at epoch {epoch + 1}")
            print("=" * 60)
            break

    # Close TensorBoard writer
    writer.close()

    # Save metrics history to CSV for analysis
    print("\n" + "=" * 60)
    print("Saving training metrics...")
    print("=" * 60)

    import csv
    metrics_csv_path = checkpoint_dir / 'training_metrics.csv'
    with open(metrics_csv_path, 'w', newline='') as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(['Epoch', 'Train_BitAccuracy', 'Train_BER', 'Train_Loss',
                             'Val_BitAccuracy', 'Val_BER', 'Val_Loss'])

        num_epochs_trained = len(metrics_history['train_bit_accuracy'])
        for i in range(num_epochs_trained):
            row = [
                i + 1,
                f"{metrics_history['train_bit_accuracy'][i]*100:.2f}",
                f"{metrics_history['train_ber'][i]:.6f}",
                f"{metrics_history['train_loss'][i]:.6f}",
            ]

            # Add validation metrics if available
            if i < len(metrics_history['val_bit_accuracy']):
                row.extend([
                    f"{metrics_history['val_bit_accuracy'][i]*100:.2f}",
                    f"{metrics_history['val_ber'][i]:.6f}",
                    f"{metrics_history['val_loss'][i]:.6f}",
                ])
            else:
                row.extend(['', '', ''])

            writer_csv.writerow(row)

    print(f"Metrics saved to: {metrics_csv_path}")

    # Generate and save training plots
    print("\nGenerating training plots...")
    plots_dir = checkpoint_dir / 'plots'
    save_training_plots(metrics_history, plots_dir)
    print(f"Plots saved to: {plots_dir}")

    # Final training summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)

    total_epochs_trained = len(metrics_history['train_bit_accuracy'])
    final_train_acc = metrics_history['train_bit_accuracy'][-1] * 100
    final_train_ber = metrics_history['train_ber'][-1]

    print(f"\nFinal Results:")
    print(f"  Epochs Trained    : {total_epochs_trained}")
    print(f"  Best Bit Accuracy : {best_bit_accuracy*100:.2f}%")
    print(f"  Final Train Acc   : {final_train_acc:.2f}%")
    print(f"  Final Train BER   : {final_train_ber:.6f}")

    if metrics_history['val_bit_accuracy']:
        final_val_acc = metrics_history['val_bit_accuracy'][-1] * 100
        final_val_ber = metrics_history['val_ber'][-1]
        print(f"  Final Val Acc     : {final_val_acc:.2f}%")
        print(f"  Final Val BER     : {final_val_ber:.6f}")

    print(f"\nModel Performance:")
    if best_bit_accuracy >= 0.95:
        print(f"  Status: EXCELLENT - Production ready")
    elif best_bit_accuracy >= 0.85:
        print(f"  Status: GOOD - Suitable for most uses")
    elif best_bit_accuracy >= 0.75:
        print(f"  Status: ACCEPTABLE - May need more training")
    else:
        print(f"  Status: POOR - Requires more training or tuning")

    print(f"\nSaved Files:")
    print(f"  Best Model     : {checkpoint_dir}/best_model_local.pth")
    print(f"  Latest Model   : {checkpoint_dir}/checkpoint_latest.pth")
    print(f"  Metrics CSV    : {metrics_csv_path}")
    print(f"  Training Plots : {plots_dir}")
    print(f"  TensorBoard    : {log_dir}")

    print(f"\nTo visualize training:")
    print(f"  tensorboard --logdir {log_dir}")

    print("=" * 60)


# =============================================================================
# Sanity Check Function
# =============================================================================
def run_sanity_check(args, device):
    """
    Run sanity check to verify model can learn on small clean dataset.

    Sanity mode:
    - 50 images only
    - Distortions OFF
    - 300-500 epochs
    - Must reach ≥95% bit accuracy

    Returns:
        bool: True if sanity check passed, False otherwise
    """
    print("\n" + "=" * 60)
    print("🔍 RUNNING SANITY CHECK")
    print("=" * 60)
    print("Testing if model can learn on small clean dataset...")
    print(f"  Dataset: 50 images")
    print(f"  Distortions: OFF")
    print(f"  Epochs: 300 (max 500 if needed)")
    print(f"  Target: ≥95% bit accuracy")
    print("=" * 60)

    # Create sanity dataset (50 images, no distortions)
    sanity_dataset = SteganographyDataset(
        image_dir=args.train_dir,
        message_length=args.message_length,
        image_size=args.image_size,
        max_images=50  # Sanity mode: only 50 images
    )

    sanity_loader = DataLoader(
        sanity_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        # Reduce workers for small dataset
        num_workers=min(args.num_workers, 2),
        pin_memory=torch.cuda.is_available()
    )

    # Initialize fresh model for sanity check
    sanity_model = StegoModel(
        message_length=args.message_length,
        image_size=args.image_size,
        enable_distortions=False  # Force distortions OFF for sanity
    )
    sanity_model = sanity_model.to(device)

    # Initialize optimizer
    sanity_optimizer = optim.Adam(
        sanity_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Create temporary writer for sanity check
    sanity_writer = SummaryWriter(
        log_dir=os.path.join(args.log_dir, 'sanity_check'))

    # Train for 300 epochs minimum, up to 500 if needed
    max_sanity_epochs = 500
    target_bit_accuracy = 0.95
    best_bit_accuracy = 0.0
    epochs_since_improvement = 0
    patience = 50  # Stop if no improvement for 50 epochs after reaching 300

    for epoch in range(max_sanity_epochs):
        # Train one epoch (no distortions, no attacks)
        metrics = train_epoch(
            sanity_model, sanity_loader, sanity_optimizer, device, epoch, sanity_writer,
            enable_distortions=False,  # Force OFF
            apply_attacks=False  # Force OFF
        )

        current_bit_accuracy = metrics['bit_accuracy']

        # Track best accuracy
        if current_bit_accuracy > best_bit_accuracy:
            best_bit_accuracy = current_bit_accuracy
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        # Print progress every 50 epochs
        if (epoch + 1) % 50 == 0:
            print(f"\nSanity Check Epoch {epoch + 1}/{max_sanity_epochs}:")
            print(f"  BER: {metrics['ber']:.6f}")
            print(f"  Bit Accuracy: {current_bit_accuracy*100:.2f}%")
            print(f"  Best: {best_bit_accuracy*100:.2f}%")

        # Check if target reached
        if current_bit_accuracy >= target_bit_accuracy:
            print("\n" + "=" * 60)
            print("[SUCCESS] SANITY CHECK PASSED!")
            print(
                f"   Reached {current_bit_accuracy*100:.2f}% bit accuracy (target: 95%)")
            print(f"   Epochs: {epoch + 1}")
            print("   Model can learn - proceeding with full training")
            print("=" * 60)
            sanity_writer.close()
            return True

        # Early stopping after 300 epochs if no improvement
        if epoch >= 300 and epochs_since_improvement >= patience:
            print("\n" + "=" * 60)
            print("[WARNING]  SANITY CHECK: Early stopping")
            print(f"   No improvement for {patience} epochs after epoch 300")
            print(f"   Best accuracy: {best_bit_accuracy*100:.2f}%")
            break

    # Sanity check failed
    print("\n" + "=" * 60)
    print("[ERROR] SANITY CHECK FAILED!")
    print(f"   Best bit accuracy: {best_bit_accuracy*100:.2f}% (target: 95%)")
    print(f"   Epochs trained: {min(epoch + 1, max_sanity_epochs)}")
    print("\n   Possible issues:")
    print("   1. Model architecture too complex for data")
    print("   2. Learning rate too high or too low")
    print("   3. Message length too large for image size")
    print("   4. Batch size too large (try batch_size=2 or 4)")
    print("\n   Suggestions:")
    print("   - Reduce message_length (try 8 or 16 bits)")
    print("   - Reduce learning_rate (try 0.00005)")
    print("   - Reduce batch_size (try 2)")
    print("   - Increase image_size (try 128 or 256)")
    print("=" * 60)
    sanity_writer.close()
    return False


# =============================================================================
# Argument Parser
# =============================================================================
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Deep Learning Steganography Model')

    # Config file argument
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config YAML file (default: config.yaml)')

    # Dataset arguments
    parser.add_argument('--train_dir', type=str, default='data/DIV2K/train',
                        help='Directory containing training images (default: data/DIV2K/train)')
    parser.add_argument('--val_dir', type=str, default=None,
                        help='Directory containing validation images')
    parser.add_argument('--dataset_type', type=str, default='auto',
                        help='Dataset type: auto, DIV2K, coco, imagenet, flat (default: auto)')
    parser.add_argument('--max_train_images', type=int, default=None,
                        help='Maximum number of BASE training images to load (not total patches)')
    parser.add_argument('--max_val_images', type=int, default=None,
                        help='Maximum number of validation images to load')
    parser.add_argument('--use_patches', action='store_true', default=False,
                        help='Use patch-based loading (extract multiple crops per image)')
    parser.add_argument('--patches_per_image', type=int, default=4,
                        help='Number of patches to extract per image (default: 4)')
    parser.add_argument('--random_crop', action='store_true', default=True,
                        help='Use random crops instead of center crops (default: True)')

    # Model arguments
    parser.add_argument('--message_length', type=int,
                        help='Length of binary message')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height = width)')
    parser.add_argument('--enable_distortions', action='store_true', default=False,
                        help='Enable distortions during training (default: False, auto-enables at 75%% accuracy)')
    parser.add_argument('--apply_attacks', action='store_true',
                        help='Apply additional attacks during training')

    # Training arguments
    parser.add_argument('--num_epochs', type=int,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float,
                        help='Weight decay for regularization')
    parser.add_argument('--num_workers', type=int,
                        help='Number of data loader workers')

    # Checkpoint arguments
    parser.add_argument('--checkpoint_dir', type=str,
                        help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str,
                        help='Directory for TensorBoard logs')
    parser.add_argument('--save_freq', type=int,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # Other arguments
    parser.add_argument('--seed', type=int,
                        help='Random seed for reproducibility')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA training')
    parser.add_argument('--sanity_mode', action='store_true',
                        help='Run sanity check: 50 images, 300-500 epochs, must reach 95%% bit accuracy')
    parser.add_argument('--skip_sanity', action='store_true',
                        help='Skip automatic sanity check before training')

    return parser.parse_args()


# =============================================================================
# Entry Point
# =============================================================================
if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()

    # Load configuration from YAML file
    print("=" * 60)
    print("Loading Configuration")
    print("=" * 60)

    config_path = args.config
    if not os.path.exists(config_path):
        print(
            f"WARNING: Config file '{config_path}' not found. Using defaults from arguments only.")
        config = {}
    else:
        print(f"Loading config from: {config_path}")
        config = load_config(config_path)

    # Merge config with command-line arguments (CLI takes precedence)
    args = merge_config_with_args(config, args)

    # Validate required arguments
    if args.train_dir is None:
        print("\nERROR: --train_dir is required")
        print("Usage: python train.py --train_dir <path_to_images>")
        print("   or: python train.py --config <path_to_config.yaml>")
        print("\nFor DIV2K: python train.py --dataset_type DIV2K --use_patches")
        print("  (defaults to data/DIV2K/train if not specified)")
        sys.exit(1)

    # Auto-detect DIV2K from path if not specified
    if args.dataset_type == 'auto' and ('div2k' in args.train_dir.lower() or 'DIV2K' in args.train_dir):
        args.dataset_type = 'DIV2K'
        print(
            f"\n[INFO] Auto-detected dataset type: DIV2K from path '{args.train_dir}'")

    # ENFORCE DIV2K TRAINING REQUIREMENTS
    if args.dataset_type == 'DIV2K':
        print("\n" + "=" * 60)
        print("DIV2K TRAINING REQUIREMENTS VALIDATION")
        print("=" * 60)

        errors = []
        warnings = []

        # 1. Message length must be 16 or 32 bits
        if args.message_length not in [16, 32]:
            errors.append(
                f"Message length must be 16 or 32 bits (got: {args.message_length})")
        else:
            print(f"[OK] Message length: {args.message_length} bits (valid)")

        # 2. Batch size must be 4
        if args.batch_size != 4:
            errors.append(
                f"Batch size must be 4 for DIV2K (got: {args.batch_size})")
        else:
            print(f"[OK] Batch size: {args.batch_size} (valid)")

        # 3. Distortions must be disabled initially
        if args.enable_distortions:
            warnings.append(
                "Distortions should be DISABLED initially (will auto-enable at 75% accuracy)")
            args.enable_distortions = False
        print(
            f"[OK] Distortions: DISABLED initially (auto-enables at 75% bit accuracy)")

        # 4. Loss weight must be 5.0 (informational only, hardcoded in training loop)
        print(f"[OK] Loss: total_loss = image_loss + 5.0 * message_loss")

        # 5. CUDA must be available
        if args.no_cuda or not torch.cuda.is_available():
            errors.append("CUDA is REQUIRED for DIV2K training")
            errors.append("  - Ensure NVIDIA GPU is available")
            errors.append("  - Install PyTorch with CUDA support")
            errors.append("  - Check: torch.cuda.is_available()")
        else:
            print(
                f"[OK] GPU: CUDA available ({torch.cuda.get_device_name(0)})")

        # Display warnings
        if warnings:
            print("\n[WARNING] WARNINGS:")
            for warning in warnings:
                print(f"   - {warning}")

        # Abort if errors
        if errors:
            print("\n" + "=" * 60)
            print("[ERROR] DIV2K TRAINING REQUIREMENTS NOT MET")
            print("=" * 60)
            print("\nERRORS:")
            for error in errors:
                print(f"   [X] {error}")
            print("\nRequired settings for DIV2K:")
            print("   • Message length: 16 or 32 bits")
            print("   • Batch size: 4")
            print("   • Distortions: DISABLED initially")
            print("   • Loss: image_loss + 5.0 * message_loss")
            print("   • GPU: CUDA required")
            print(
                "\nUse config files: config_div2k_quick.yaml or config_div2k_balanced.yaml")
            print("=" * 60)
            sys.exit(1)

        print("\n[SUCCESS] All DIV2K requirements satisfied")
        print("=" * 60)

    # Print final configuration
    print("\n" + "=" * 60)
    print("Final Configuration")
    print("=" * 60)
    print_config(config)

    # Determine device before sanity check
    if args.no_cuda or not torch.cuda.is_available():
        device = torch.device('cpu')
        if not args.no_cuda:
            print("\n[WARNING]  WARNING: CUDA not available, using CPU")
    else:
        device = torch.device('cuda')

    # Run sanity check if not skipped and not in sanity-only mode
    if not args.skip_sanity and not args.sanity_mode:
        print("\n" + "=" * 60)
        print("AUTOMATIC SANITY CHECK")
        print("=" * 60)
        print("Verifying model can learn before full training...")
        print("(Use --skip_sanity to disable this check)")

        sanity_passed = run_sanity_check(args, device)

        if not sanity_passed:
            print("\n[ERROR] Aborting training due to failed sanity check.")
            print("   Fix the issues above or use --skip_sanity to proceed anyway.")
            sys.exit(1)

        print("\n" + "=" * 60)
        print("Proceeding with full training...")
        print("=" * 60)

    # If sanity-only mode, run sanity check and exit
    if args.sanity_mode:
        sanity_passed = run_sanity_check(args, device)
        sys.exit(0 if sanity_passed else 1)

    # Start normal training
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    print(f"\n>>> Training begins now...\n")
    main(args)
