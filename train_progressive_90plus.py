"""
Progressive Multi-Phase Training Script for 90%+ Accuracy

This script implements a 6-phase progressive training strategy:
1. Clean Training (Target: >94% accuracy)
2. JPEG Compression (Target: >90%)
3. Add Gaussian Blur (Target: >90%)
4. Add Resize Attack (Target: >90%)
5. Add Color Jitter (Target: >90%)
6. Combined Attacks (Target: >88%)

Features:
- Cyclical Learning Rate (CyclicLR)
- Expanded dataset (400 DIV2K images)
- Pixel delta monitoring (<0.02)
- Auto-checkpoint loading between phases
- Early stopping per phase
- Comprehensive evaluation
"""

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
import numpy as np

# Import project modules
from models.model import StegoModel
from attacks import JPEGCompression, GaussianNoise, ResizeAttack, ColorJitter
from attacks.blur import GaussianBlur


# =============================================================================
# Dataset Class with 400 Image Support
# =============================================================================
class SteganographyDataset(Dataset):
    """Dataset for loading images with support for 400+ images"""
    
    def __init__(self, image_dir, message_length=16, image_size=128, 
                 max_images=400, use_patches=True, patches_per_image=4):
        """
        Args:
            image_dir: Directory containing images
            message_length: Length of binary messages (default: 16 bits)
            image_size: Size to crop images to (default: 128)
            max_images: Maximum number of images (default: 400)
            use_patches: Extract multiple patches per image
            patches_per_image: Number of patches per image
        """
        self.image_dir = image_dir
        self.message_length = message_length
        self.image_size = image_size
        self.use_patches = use_patches
        self.patches_per_image = patches_per_image
        
        # Find all images
        self.image_paths = []
        for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
            self.image_paths.extend(glob.glob(
                os.path.join(image_dir, '**', f'*{ext}'), recursive=True))
        
        # Limit to max_images
        if max_images and max_images < len(self.image_paths):
            import random
            random.seed(42)  # For reproducibility
            random.shuffle(self.image_paths)
            self.image_paths = self.image_paths[:max_images]
        
        print(f"✓ Loaded {len(self.image_paths)} images from {image_dir}")
        
        # Calculate total samples
        if use_patches:
            self.total_samples = len(self.image_paths) * patches_per_image
            print(f"✓ Total training samples (with patches): {self.total_samples}")
        else:
            self.total_samples = len(self.image_paths)
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.RandomCrop(image_size) if use_patches else transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        if self.use_patches:
            image_idx = idx // self.patches_per_image
        else:
            image_idx = idx
        
        try:
            image_path = self.image_paths[image_idx]
            image = Image.open(image_path).convert('RGB')
            
            # Ensure image is large enough for patching
            if self.use_patches:
                width, height = image.size
                if width < self.image_size or height < self.image_size:
                    scale = max(self.image_size / width, self.image_size / height)
                    new_width = int(width * scale) + 1
                    new_height = int(height * scale) + 1
                    image = image.resize((new_width, new_height), Image.BICUBIC)
            
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading {self.image_paths[image_idx]}: {e}")
            return self.__getitem__((idx + 1) % len(self))
        
        # Generate random binary message
        message = torch.randint(0, 2, (self.message_length,)).float()
        return image, message


# =============================================================================
# Training Function with Phase Support
# =============================================================================
def train_epoch_phase(model, dataloader, optimizer, scheduler, device, epoch, 
                     phase_config, writer, global_step, args, use_amp=True):
    """
    Train for one epoch with specific phase configuration.
    
    Args:
        model: StegoModel instance
        dataloader: Training data loader
        optimizer: Optimizer
        scheduler: CyclicLR scheduler
        device: Training device
        epoch: Current epoch number
        phase_config: Dictionary containing phase settings
        writer: TensorBoard writer
        global_step: Global training step counter
        use_amp: Use automatic mixed precision
    
    Returns:
        Dictionary of metrics and updated global_step
    """
    model.train()
    scaler = GradScaler(enabled=use_amp and device.type == 'cuda')
    
    # Extract phase configuration
    phase_name = phase_config['name']
    alpha = phase_config['alpha']
    beta = phase_config['beta']
    attacks = phase_config['attacks']
    
    # Initialize attack modules
    jpeg = JPEGCompression(quality_range=(70, 95)).to(device) if 'jpeg' in attacks else None
    blur = GaussianBlur(kernel_size_range=(3, 5), sigma_range=(0.3, 1.2)).to(device) if 'blur' in attacks else None
    resize = ResizeAttack(scale_range=(0.6, 0.9)).to(device) if 'resize' in attacks else None
    color_jitter = ColorJitter().to(device) if 'color_jitter' in attacks else None
    
    total_loss = 0.0
    total_image_loss = 0.0
    total_message_loss = 0.0
    total_bit_accuracy = 0.0
    total_pixel_delta = 0.0
    
    print(f"\nEpoch {epoch}/{args.max_epochs_per_phase} - {phase_name}")
    print("-" * 80)
    
    for batch_idx, (cover_images, binary_messages) in enumerate(dataloader):
        cover_images = cover_images.to(device, non_blocking=True)
        binary_messages = binary_messages.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        with autocast(enabled=use_amp and device.type == 'cuda'):
            # Forward pass through encoder
            stego_images = model.encode(cover_images, binary_messages)
            distorted_stego = stego_images.clone()
            
            # Apply attacks based on phase configuration
            if 'jpeg' in attacks and torch.rand(1).item() < attacks['jpeg']:
                distorted_stego = jpeg(distorted_stego)
            
            if 'blur' in attacks and torch.rand(1).item() < attacks['blur']:
                distorted_stego = blur(distorted_stego)
            
            if 'resize' in attacks and torch.rand(1).item() < attacks['resize']:
                distorted_stego = resize(distorted_stego)
            
            if 'color_jitter' in attacks and torch.rand(1).item() < attacks['color_jitter']:
                distorted_stego = color_jitter(distorted_stego)
            
            # Apply all attacks simultaneously if combined mode
            if phase_config.get('combined_mode', False) and torch.rand(1).item() < 0.7:
                distorted_stego = jpeg(distorted_stego) if jpeg else distorted_stego
                distorted_stego = blur(distorted_stego) if blur else distorted_stego
                distorted_stego = resize(distorted_stego) if resize else distorted_stego
                distorted_stego = color_jitter(distorted_stego) if color_jitter else distorted_stego
            
            # Decode message
            decoded_logits = model.decode(distorted_stego, return_logits=True)
            decoded_message = (torch.sigmoid(decoded_logits) > 0.5).float()
            
            # Compute losses with phase-specific weights
            image_loss = nn.functional.mse_loss(stego_images, cover_images)
            message_loss = nn.functional.binary_cross_entropy_with_logits(
                decoded_logits, binary_messages)
            total_loss_batch = alpha * image_loss + beta * message_loss
            
            # Compute metrics
            bit_accuracy = (decoded_message == binary_messages).float().mean()
            pixel_delta = torch.abs(stego_images - cover_images).mean()
        
        # Backward pass with gradient scaling
        scaler.scale(total_loss_batch).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        # Step the cyclical learning rate scheduler AFTER optimizer step
        scheduler.step()
        
        # Accumulate metrics
        total_loss += total_loss_batch.item()
        total_image_loss += image_loss.item()
        total_message_loss += message_loss.item()
        total_bit_accuracy += bit_accuracy.item()
        total_pixel_delta += pixel_delta.item()
        
        # Print batch progress every 20 batches
        if batch_idx % 20 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            avg_acc = total_bit_accuracy / (batch_idx + 1)
            avg_delta = total_pixel_delta / (batch_idx + 1)
            print(f"  Batch [{batch_idx:3d}/{len(dataloader)}] "
                  f"Loss: {total_loss_batch.item():.4f} | "
                  f"Acc: {bit_accuracy.item()*100:5.2f}% | "
                  f"PixΔ: {pixel_delta.item():.5f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Log to TensorBoard
        if batch_idx % 10 == 0:
            writer.add_scalar(f'{phase_name}/BatchLoss', total_loss_batch.item(), global_step)
            writer.add_scalar(f'{phase_name}/BatchAccuracy', bit_accuracy.item(), global_step)
            writer.add_scalar(f'{phase_name}/LearningRate', optimizer.param_groups[0]['lr'], global_step)
        
        global_step += 1
    
    # Calculate epoch averages
    num_batches = len(dataloader)
    metrics = {
        'loss': total_loss / num_batches,
        'image_loss': total_image_loss / num_batches,
        'message_loss': total_message_loss / num_batches,
        'bit_accuracy': total_bit_accuracy / num_batches,
        'pixel_delta': total_pixel_delta / num_batches,
    }
    
    # Log epoch metrics
    writer.add_scalar(f'{phase_name}/EpochLoss', metrics['loss'], epoch)
    writer.add_scalar(f'{phase_name}/EpochAccuracy', metrics['bit_accuracy'], epoch)
    writer.add_scalar(f'{phase_name}/PixelDelta', metrics['pixel_delta'], epoch)
    
    # Print detailed epoch summary
    print(f"\n{'='*70}")
    print(f"EPOCH {epoch} SUMMARY - {phase_name}")
    print(f"{'='*70}")
    print(f"  Loss:           {metrics['loss']:.6f}")
    print(f"  Image Loss:     {metrics['image_loss']:.6f}")
    print(f"  Message Loss:   {metrics['message_loss']:.6f}")
    print(f"  Bit Accuracy:   {metrics['bit_accuracy']*100:.2f}%")
    print(f"  Pixel Delta:    {metrics['pixel_delta']:.6f}")
    print(f"  Learning Rate:  {optimizer.param_groups[0]['lr']:.2e}")
    print(f"{'='*70}\n")
    
    return metrics, global_step


# =============================================================================
# Validation Function
# =============================================================================
def validate_phase(model, dataloader, device, phase_config):
    """Validate model on clean or attacked images"""
    model.eval()
    
    attacks = phase_config['attacks']
    jpeg = JPEGCompression(quality_range=(70, 95)).to(device) if 'jpeg' in attacks else None
    blur = GaussianBlur(kernel_size_range=(3, 5), sigma_range=(0.3, 1.2)).to(device) if 'blur' in attacks else None
    resize = ResizeAttack(scale_range=(0.6, 0.9)).to(device) if 'resize' in attacks else None
    color_jitter = ColorJitter().to(device) if 'color_jitter' in attacks else None
    
    total_bit_accuracy = 0.0
    total_pixel_delta = 0.0
    
    with torch.no_grad():
        for cover_images, binary_messages in dataloader:
            cover_images = cover_images.to(device)
            binary_messages = binary_messages.to(device)
            
            # Encode
            stego_images = model.encode(cover_images, binary_messages)
            distorted_stego = stego_images.clone()
            
            # Apply attacks
            if 'jpeg' in attacks and attacks['jpeg'] > 0:
                distorted_stego = jpeg(distorted_stego)
            if 'blur' in attacks and attacks['blur'] > 0:
                distorted_stego = blur(distorted_stego)
            if 'resize' in attacks and attacks['resize'] > 0:
                distorted_stego = resize(distorted_stego)
            if 'color_jitter' in attacks and attacks['color_jitter'] > 0:
                distorted_stego = color_jitter(distorted_stego)
            
            # Decode
            decoded_logits = model.decode(distorted_stego, return_logits=True)
            decoded_message = (torch.sigmoid(decoded_logits) > 0.5).float()
            
            # Metrics
            bit_accuracy = (decoded_message == binary_messages).float().mean()
            pixel_delta = (stego_images - cover_images).abs().mean()
            
            total_bit_accuracy += bit_accuracy.item()
            total_pixel_delta += pixel_delta.item()
    
    return {
        'bit_accuracy': total_bit_accuracy / len(dataloader),
        'pixel_delta': total_pixel_delta / len(dataloader)
    }


# =============================================================================
# Progressive Training Pipeline
# =============================================================================
def train_progressive(args):
    """
    Main progressive training pipeline with 6 phases.
    """
    print("\n" + "="*80)
    print("🚀 PROGRESSIVE MULTI-PHASE TRAINING FOR 90%+ ACCURACY")
    print("="*80 + "\n")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Device: {device}")
    
    # Create directories
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir) / datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    # Load datasets (400 images)
    print(f"\n📁 Loading dataset from: {args.train_dir}")
    train_dataset = SteganographyDataset(
        args.train_dir,
        message_length=args.message_length,
        image_size=args.image_size,
        max_images=args.max_images,
        use_patches=True,
        patches_per_image=4
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Validation set (separate 50 images)
    if args.val_dir:
        val_dataset = SteganographyDataset(
            args.val_dir,
            message_length=args.message_length,
            image_size=args.image_size,
            max_images=50,
            use_patches=False
        )
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    else:
        val_loader = None
    
    # Define 6 training phases
    phases = [
        {
            'id': 1,
            'name': 'PHASE1_CLEAN',
            'description': 'Clean Training',
            'target_accuracy': 0.94,
            'alpha': 1.0,
            'beta': 2.0,
            'attacks': {},  # No attacks
            'combined_mode': False,
            'checkpoint': 'model_phase1_clean_94.pth'
        },
        {
            'id': 2,
            'name': 'PHASE2_JPEG',
            'description': 'JPEG Compression',
            'target_accuracy': 0.90,
            'alpha': 0.9,
            'beta': 2.0,
            'attacks': {'jpeg': 0.5},
            'combined_mode': False,
            'checkpoint': 'model_phase2_jpeg_90.pth'
        },
        {
            'id': 3,
            'name': 'PHASE3_BLUR',
            'description': 'JPEG + Gaussian Blur',
            'target_accuracy': 0.90,
            'alpha': 0.9,
            'beta': 2.2,
            'attacks': {'jpeg': 0.3, 'blur': 0.5},
            'combined_mode': False,
            'checkpoint': 'model_phase3_blur_90.pth'
        },
        {
            'id': 4,
            'name': 'PHASE4_RESIZE',
            'description': 'JPEG + Blur + Resize',
            'target_accuracy': 0.90,
            'alpha': 0.85,
            'beta': 2.5,
            'attacks': {'jpeg': 0.25, 'blur': 0.35, 'resize': 0.5},
            'combined_mode': False,
            'checkpoint': 'model_phase4_resize_90.pth'
        },
        {
            'id': 5,
            'name': 'PHASE5_ALL',
            'description': 'All Individual Attacks',
            'target_accuracy': 0.90,
            'alpha': 0.8,
            'beta': 2.5,
            'attacks': {'jpeg': 0.25, 'blur': 0.3, 'resize': 0.35, 'color_jitter': 0.4},
            'combined_mode': False,
            'checkpoint': 'model_phase5_all_90.pth'
        },
        {
            'id': 6,
            'name': 'PHASE6_COMBINED',
            'description': 'Combined Attacks',
            'target_accuracy': 0.88,
            'alpha': 0.8,
            'beta': 3.0,
            'attacks': {'jpeg': 1.0, 'blur': 1.0, 'resize': 1.0, 'color_jitter': 1.0},
            'combined_mode': True,
            'checkpoint': 'model_phase6_combined_88.pth'
        }
    ]
    
    # Start progressive training
    global_step = 0
    
    for phase in phases:
        print("\n" + "="*80)
        print(f"🎯 PHASE {phase['id']}/6: {phase['description'].upper()} (Target: >{phase['target_accuracy']*100:.0f}%)")
        print("="*80)
        print(f"Configuration:")
        print(f"  • Alpha (image loss weight): {phase['alpha']}")
        print(f"  • Beta (message loss weight): {phase['beta']}")
        print(f"  • Attacks: {phase['attacks']}")
        print(f"  • Min Epochs: {args.min_epochs_per_phase}")
        print(f"  • Max Epochs: {args.max_epochs_per_phase}")
        print()
        
        # Initialize or load model
        if phase['id'] == 1:
            # Phase 1: Try to load best existing model, otherwise initialize new
            best_model_path = Path('checkpoints/model_IMPROVED_BLUR.pth')
            if best_model_path.exists():
                print(f"✓ Loading best existing model: {best_model_path}")
                model = StegoModel(message_length=args.message_length, image_size=args.image_size).to(device)
                checkpoint = torch.load(best_model_path, map_location=device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"  Previous accuracy: {checkpoint.get('accuracy', 'N/A')}")
                else:
                    model.load_state_dict(checkpoint)
                print("✓ Continuing training from best checkpoint")
            else:
                model = StegoModel(message_length=args.message_length, image_size=args.image_size).to(device)
                print("✓ Initialized new model")
        else:
            # Load previous phase checkpoint
            prev_checkpoint = checkpoint_dir / phases[phase['id']-2]['checkpoint']
            if prev_checkpoint.exists():
                checkpoint = torch.load(prev_checkpoint, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✓ Loaded checkpoint from Phase {phase['id']-1}: {prev_checkpoint.name}")
            else:
                print(f"⚠ Warning: Previous checkpoint not found at {prev_checkpoint}")
                print("  Continuing with current model state")
        
        # Setup optimizer with Cyclical Learning Rate
        optimizer = optim.Adam(model.parameters(), lr=args.base_lr)
        
        # CyclicLR scheduler
        scheduler = optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=args.base_lr,
            max_lr=args.max_lr,
            step_size_up=args.cyclic_step_size,
            mode='triangular2',
            cycle_momentum=False
        )
        
        print(f"✓ Optimizer: Adam with CyclicLR")
        print(f"  • Base LR: {args.base_lr:.2e}")
        print(f"  • Max LR: {args.max_lr:.2e}")
        print(f"  • Step size: {args.cyclic_step_size}")
        print()
        
        # Training loop for this phase
        best_accuracy = 0.0
        epochs_without_improvement = 0
        target_reached = False
        
        for epoch in range(1, args.max_epochs_per_phase + 1):
            # Train one epoch
            metrics, global_step = train_epoch_phase(
                model, train_loader, optimizer, scheduler, device,
                epoch, phase, writer, global_step, args, use_amp=args.use_amp
            )
            
            # Validate if validation set provided
            if val_loader:
                val_metrics = validate_phase(model, val_loader, device, phase)
                print(f"  Validation → Accuracy: {val_metrics['bit_accuracy']*100:.2f}%, "
                      f"Pixel Δ: {val_metrics['pixel_delta']:.4f}")
            
            # Check pixel delta constraint
            if metrics['pixel_delta'] > args.max_pixel_delta:
                print(f"  ⚠ Pixel delta ({metrics['pixel_delta']:.4f}) > threshold ({args.max_pixel_delta})")
                phase['alpha'] += 0.1
                print(f"  → Increased alpha to {phase['alpha']:.2f}")
            
            # Check if target reached
            if metrics['bit_accuracy'] >= phase['target_accuracy']:
                if epoch >= args.min_epochs_per_phase:
                    print(f"  ✅ Target accuracy {phase['target_accuracy']*100:.0f}% reached!")
                    target_reached = True
                else:
                    print(f"  🎯 Target reached, but continuing to min epoch {args.min_epochs_per_phase}")
            
            # Save best model
            if metrics['bit_accuracy'] > best_accuracy:
                best_accuracy = metrics['bit_accuracy']
                epochs_without_improvement = 0
                
                checkpoint_path = checkpoint_dir / phase['checkpoint']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': best_accuracy,
                    'pixel_delta': metrics['pixel_delta'],
                    'phase': phase['id']
                }, checkpoint_path)
                print(f"  💾 Saved checkpoint: {checkpoint_path.name}")
            else:
                epochs_without_improvement += 1
            
            # Early stopping check
            if target_reached and epochs_without_improvement >= args.early_stopping_patience:
                print(f"\n  ⏹ Early stopping: No improvement for {args.early_stopping_patience} epochs")
                break
        
        # Phase complete
        print(f"\n✅ Phase {phase['id']} Complete!")
        print(f"   • Best Accuracy: {best_accuracy*100:.2f}%")
        print(f"   • Target: {phase['target_accuracy']*100:.0f}%")
        print(f"   • Status: {'✓ PASSED' if best_accuracy >= phase['target_accuracy'] else '✗ FAILED'}")
        print()
    
    # Final evaluation
    print("\n" + "="*80)
    print("📊 FINAL EVALUATION")
    print("="*80 + "\n")
    
    # Load final model
    final_checkpoint = checkpoint_dir / phases[-1]['checkpoint']
    if final_checkpoint.exists():
        checkpoint = torch.load(final_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded final model: {final_checkpoint.name}\n")
    
    # Evaluate on each attack type
    eval_configs = [
        {'name': 'Clean (No Attack)', 'attacks': {}},
        {'name': 'JPEG Compression', 'attacks': {'jpeg': 1.0}},
        {'name': 'Gaussian Blur', 'attacks': {'blur': 1.0}},
        {'name': 'Resize Attack', 'attacks': {'resize': 1.0}},
        {'name': 'Color Jitter', 'attacks': {'color_jitter': 1.0}},
        {'name': 'ALL COMBINED', 'attacks': {'jpeg': 1.0, 'blur': 1.0, 'resize': 1.0, 'color_jitter': 1.0}, 'combined_mode': True},
    ]
    
    print("Final Model Performance:")
    print("-" * 60)
    
    for eval_config in eval_configs:
        if val_loader:
            eval_metrics = validate_phase(model, val_loader, device, eval_config)
            print(f"{eval_config['name']:25} → Accuracy: {eval_metrics['bit_accuracy']*100:.2f}%, "
                  f"Pixel Δ: {eval_metrics['pixel_delta']:.4f}")
    
    print("\n" + "="*80)
    print("🎉 PROGRESSIVE TRAINING COMPLETE!")
    print("="*80)
    print(f"\nCheckpoints saved in: {checkpoint_dir}")
    print(f"TensorBoard logs saved in: {log_dir}")
    print(f"\nTo view training progress:")
    print(f"  tensorboard --logdir {log_dir}")
    
    writer.close()


# =============================================================================
# Main Entry Point
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Progressive Multi-Phase Training for 90%+ Accuracy')
    
    # Dataset arguments
    parser.add_argument('--train_dir', type=str, default='data/DIV2K/train',
                        help='Training images directory')
    parser.add_argument('--val_dir', type=str, default=None,
                        help='Validation images directory (optional)')
    parser.add_argument('--max_images', type=int, default=400,
                        help='Maximum number of training images (default: 400)')
    
    # Model arguments
    parser.add_argument('--message_length', type=int, default=16,
                        help='Binary message length (default: 16 bits)')
    parser.add_argument('--image_size', type=int, default=128,
                        help='Image size (default: 128)')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size (default: 8)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    parser.add_argument('--min_epochs_per_phase', type=int, default=20,
                        help='Minimum epochs per phase (default: 20)')
    parser.add_argument('--max_epochs_per_phase', type=int, default=100,
                        help='Maximum epochs per phase (default: 100)')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help='Early stopping patience (default: 10)')
    
    # Learning rate arguments
    parser.add_argument('--base_lr', type=float, default=1e-4,
                        help='Base learning rate for CyclicLR (default: 1e-4)')
    parser.add_argument('--max_lr', type=float, default=5e-3,
                        help='Maximum learning rate for CyclicLR (default: 5e-3)')
    parser.add_argument('--cyclic_step_size', type=int, default=200,
                        help='Step size for CyclicLR (default: 200)')
    
    # Constraints
    parser.add_argument('--max_pixel_delta', type=float, default=0.02,
                        help='Maximum allowed pixel delta (default: 0.02)')
    
    # Other arguments
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='runs',
                        help='Directory for TensorBoard logs')
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use automatic mixed precision (default: True)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Start progressive training
    train_progressive(args)


if __name__ == '__main__':
    main()
