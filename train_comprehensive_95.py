"""
Comprehensive Training for All Transformations with DIV2K Dataset
Target: 95%+ accuracy AND pixel delta ≈ 0.02
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from models.model import StegoModel
from PIL import Image
import os
import glob


class DIV2KDataset(Dataset):
    """DIV2K Dataset loader."""

    def __init__(self, image_dir, message_length, image_size=128, max_images=200):
        self.message_length = message_length
        self.image_size = image_size

        # Find all images
        self.image_paths = []
        for ext in ['.png', '.jpg', '.jpeg']:
            self.image_paths.extend(glob.glob(os.path.join(
                image_dir, '**', f'*{ext}'), recursive=True))

        # Limit number of images
        if max_images and len(self.image_paths) > max_images:
            self.image_paths = self.image_paths[:max_images]

        # Transform
        self.transform = transforms.Compose([
            transforms.RandomCrop(image_size),
            transforms.ToTensor(),
        ])

        print(f"  Loaded {len(self.image_paths)} DIV2K images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        # Apply transform
        image = self.transform(image)

        # Generate random binary message
        message = torch.randint(0, 2, (self.message_length,)).float()

        return image, message


def calculate_psnr(img1, img2):
    """Calculate PSNR between two images."""
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse < 1e-10:
        return 100.0
    return 20 * torch.log10(torch.tensor(1.0 / (mse ** 0.5))).item()


def train_comprehensive(model, dataloader, optimizer, scheduler, device,
                        phase_name, current_phase, total_phases,
                        max_epochs=30, target_accuracy=0.95, target_pixel_delta_range=(0.018, 0.022)):
    """Train with comprehensive distortions."""

    print(f"\n{'='*80}")
    print(f"PHASE {current_phase}/{total_phases}: {phase_name}")
    print(f"Target: Accuracy ≥ {target_accuracy*100}% AND Pixel Delta ≈ 0.02")
    print(f"{'='*80}\n")

    best_accuracy = 0
    best_pixel_delta = float('inf')

    for epoch in range(max_epochs):
        model.train()

        epoch_loss = 0
        epoch_img_loss = 0
        epoch_msg_loss = 0
        epoch_accuracy = 0
        epoch_pixel_delta = 0
        epoch_psnr = 0
        num_batches = 0

        print(f"\n{'─'*80}")
        print(
            f"Epoch {epoch+1}/{max_epochs} (Phase {current_phase}/{total_phases})")
        print(f"{'─'*80}")

        for batch_idx, (cover_images, binary_messages) in enumerate(dataloader):
            cover_images = cover_images.to(device)
            binary_messages = binary_messages.to(device)

            # Forward pass
            optimizer.zero_grad()
            loss_dict = model.compute_loss(
                cover_images, binary_messages, alpha=1.0, beta=2.0)

            # Backward pass
            loss_dict['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Calculate metrics without distortions for clean image quality
            with torch.no_grad():
                outputs = model(cover_images, binary_messages,
                                apply_distortions=False)
                stego = outputs['stego_image']
                pixel_delta = torch.mean(
                    torch.abs(stego - cover_images)).item()
                psnr = calculate_psnr(cover_images, stego)

            accuracy = loss_dict['bit_accuracy'].item()
            ber = loss_dict['ber'].item()

            epoch_loss += loss_dict['total_loss'].item()
            epoch_img_loss += loss_dict['image_loss'].item()
            epoch_msg_loss += loss_dict['message_loss'].item()
            epoch_accuracy += accuracy
            epoch_pixel_delta += pixel_delta
            epoch_psnr += psnr
            num_batches += 1

            # Print every 5 batches
            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(dataloader):
                print(f"  Batch {batch_idx+1:3d}/{len(dataloader):3d} | "
                      f"Loss: {loss_dict['total_loss'].item():.4f} | "
                      f"Acc: {accuracy*100:6.2f}% | "
                      f"BER: {ber*100:5.2f}% | "
                      f"Δ: {pixel_delta:.6f} | "
                      f"PSNR: {psnr:5.2f}dB")

        # Epoch summary
        avg_loss = epoch_loss / num_batches
        avg_img_loss = epoch_img_loss / num_batches
        avg_msg_loss = epoch_msg_loss / num_batches
        avg_accuracy = epoch_accuracy / num_batches
        avg_pixel_delta = epoch_pixel_delta / num_batches
        avg_psnr = epoch_psnr / num_batches
        avg_ber = 1 - avg_accuracy

        print(f"\n  {'EPOCH SUMMARY':^76}")
        print(f"  {'-'*76}")
        print(
            f"  Total Loss: {avg_loss:.6f} | Img Loss: {avg_img_loss:.6f} | Msg Loss: {avg_msg_loss:.6f}")
        print(
            f"  Accuracy:   {avg_accuracy*100:6.2f}% | BER: {avg_ber*100:5.2f}%")
        print(f"  Pixel Δ:    {avg_pixel_delta:.6f} | PSNR: {avg_psnr:.2f} dB")

        # Update learning rate
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  LR: {current_lr:.2e}")

        # Check if targets are met
        pixel_in_range = target_pixel_delta_range[0] <= avg_pixel_delta <= target_pixel_delta_range[1]
        target_met = avg_accuracy >= target_accuracy and pixel_in_range

        # Track best
        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
        if avg_pixel_delta < best_pixel_delta:
            best_pixel_delta = avg_pixel_delta

        print(
            f"  Best Acc: {best_accuracy*100:6.2f}% | Best Δ: {best_pixel_delta:.6f}")

        if target_met:
            print(
                f"\n  ✓✓✓ TARGET ACHIEVED! Acc: {avg_accuracy*100:.2f}%, Δ: {avg_pixel_delta:.6f} ✓✓✓")
            return True, avg_accuracy, avg_pixel_delta

        # Progress indicator
        if avg_accuracy >= target_accuracy:
            print(f"  ✓ Accuracy target met! Working on pixel delta...")
        elif pixel_in_range:
            print(f"  ✓ Pixel delta in range! Working on accuracy...")

    print(
        f"\n  ✗ Max epochs reached. Final: Acc {avg_accuracy*100:.2f}%, Δ {avg_pixel_delta:.6f}")
    return False, avg_accuracy, avg_pixel_delta


def main():
    print(f"\n{'#'*80}")
    print(f"#{'COMPREHENSIVE STEGANOGRAPHY TRAINING - DIV2K':^78}#")
    print(f"#{'Target: 95%+ Accuracy & Pixel Delta ≈ 0.02':^78}#")
    print(f"{'#'*80}\n")

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")

    # Configuration
    message_length = 16
    image_size = 128
    batch_size = 4
    num_images = 200
    max_epochs = 30
    div2k_path = 'data/DIV2K/train'

    print(f"Configuration:")
    print(f"  Message Length: {message_length} bits")
    print(f"  Image Size: {image_size}x{image_size}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Max Images: {num_images}")
    print(f"  Max Epochs per Phase: {max_epochs}")
    print(f"  Dataset: DIV2K")

    # Load DIV2K dataset
    print(f"\nLoading DIV2K dataset from {div2k_path}...")
    dataset = DIV2KDataset(div2k_path, message_length,
                           image_size, max_images=num_images)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=2)
    print(f"✓ Dataset ready\n")

    # Load model
    print(f"Loading model...")
    if os.path.exists('checkpoints/model_with_blur.pth'):
        checkpoint_path = 'checkpoints/model_with_blur.pth'
        print(f"  From: {checkpoint_path}")
    else:
        checkpoint_path = 'checkpoints/best_model_local.pth'
        print(f"  From: {checkpoint_path} (base model)")

    model = StegoModel(message_length, image_size, enable_distortions=True)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print(f"✓ Model loaded\n")

    total_phases = 4
    results = {}

    # ============================================================================
    # PHASE 1: Clean Encoding/Decoding (No distortions)
    # ============================================================================
    print(f"\n{'#'*80}")
    print(f"# PHASE 1/4: CLEAN ENCODING/DECODING (No Distortions)")
    print(f"{'#'*80}")

    # Disable all distortions
    def forward_clean(images, apply_all=False, jpeg_only=False):
        return images

    model.distortions.forward = forward_clean

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.5, patience=3)

    success1, acc1, delta1 = train_comprehensive(
        model, dataloader, optimizer, scheduler, device,
        "Clean Encoding/Decoding", 1, total_phases, max_epochs
    )
    results['clean'] = (success1, acc1, delta1)

    # Save
    torch.save({
        'model_state_dict': model.state_dict(),
        'accuracy': acc1,
        'pixel_delta': delta1,
        'phase': 'clean'
    }, 'checkpoints/model_clean_95_DIV2K.pth')
    print(f"\n✓ Saved: checkpoints/model_clean_95_DIV2K.pth")

    # ============================================================================
    # PHASE 2: + JPEG Compression
    # ============================================================================
    print(f"\n{'#'*80}")
    print(f"# PHASE 2/4: JPEG COMPRESSION")
    print(f"{'#'*80}")

    def forward_jpeg(images, apply_all=False, jpeg_only=False):
        if model.distortions.training:
            images = model.distortions.apply_jpeg_compression(
                images, probability=0.5)
        return images

    model.distortions.forward = forward_jpeg

    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.5, patience=3)

    success2, acc2, delta2 = train_comprehensive(
        model, dataloader, optimizer, scheduler, device,
        "JPEG Compression", 2, total_phases, max_epochs
    )
    results['jpeg'] = (success2, acc2, delta2)

    # Save
    torch.save({
        'model_state_dict': model.state_dict(),
        'accuracy': acc2,
        'pixel_delta': delta2,
        'phase': 'jpeg'
    }, 'checkpoints/model_jpeg_95_DIV2K.pth')
    print(f"\n✓ Saved: checkpoints/model_jpeg_95_DIV2K.pth")

    # ============================================================================
    # PHASE 3: + Gaussian Blur + Resize Attack
    # ============================================================================
    print(f"\n{'#'*80}")
    print(f"# PHASE 3/4: GAUSSIAN BLUR + RESIZE ATTACK")
    print(f"{'#'*80}")

    def forward_blur_resize(images, apply_all=False, jpeg_only=False):
        if model.distortions.training:
            images = model.distortions.apply_jpeg_compression(
                images, probability=0.3)
            images = model.distortions.apply_gaussian_blur_attack(
                images, probability=0.4)
            images = model.distortions.apply_resize_attack(
                images, probability=0.4)
        return images

    model.distortions.forward = forward_blur_resize

    optimizer = optim.Adam(model.parameters(), lr=3e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.5, patience=3)

    success3, acc3, delta3 = train_comprehensive(
        model, dataloader, optimizer, scheduler, device,
        "Blur + Resize", 3, total_phases, max_epochs
    )
    results['blur_resize'] = (success3, acc3, delta3)

    # Save
    torch.save({
        'model_state_dict': model.state_dict(),
        'accuracy': acc3,
        'pixel_delta': delta3,
        'phase': 'blur_resize'
    }, 'checkpoints/model_blur_resize_95_DIV2K.pth')
    print(f"\n✓ Saved: checkpoints/model_blur_resize_95_DIV2K.pth")

    # ============================================================================
    # PHASE 4: + Color Jitter (ALL TRANSFORMATIONS)
    # ============================================================================
    print(f"\n{'#'*80}")
    print(f"# PHASE 4/4: ALL TRANSFORMATIONS (JPEG + Blur + Resize + Color Jitter)")
    print(f"{'#'*80}")

    def forward_all(images, apply_all=False, jpeg_only=False):
        if model.distortions.training:
            images = model.distortions.apply_jpeg_compression(
                images, probability=0.3)
            images = model.distortions.apply_gaussian_blur_attack(
                images, probability=0.3)
            images = model.distortions.apply_resize_attack(
                images, probability=0.3)
            images = model.distortions.apply_color_jitter_attack(
                images, probability=0.3)
        return images

    model.distortions.forward = forward_all

    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.5, patience=3)

    success4, acc4, delta4 = train_comprehensive(
        model, dataloader, optimizer, scheduler, device,
        "ALL TRANSFORMATIONS", 4, total_phases, max_epochs
    )
    results['all'] = (success4, acc4, delta4)

    # Save final
    torch.save({
        'model_state_dict': model.state_dict(),
        'accuracy': acc4,
        'pixel_delta': delta4,
        'phase': 'all_transformations'
    }, 'checkpoints/model_ALL_TRANSFORMATIONS_95_DIV2K.pth')
    print(f"\n✓ Saved: checkpoints/model_ALL_TRANSFORMATIONS_95_DIV2K.pth")

    # ============================================================================
    # FINAL SUMMARY
    # ============================================================================
    print(f"\n{'='*80}")
    print(f"{'FINAL TRAINING SUMMARY':^80}")
    print(f"{'='*80}\n")

    print(f"{'Phase':<30} {'Accuracy':<12} {'Pixel Δ':<12} {'Status'}")
    print(f"{'-'*80}")

    for phase_name, (success, acc, delta) in [
        ("Clean Encoding/Decoding", results['clean']),
        ("+ JPEG Compression", results['jpeg']),
        ("+ Blur + Resize", results['blur_resize']),
        ("+ Color Jitter (ALL)", results['all'])
    ]:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{phase_name:<30} {acc*100:>6.2f}%     {delta:>8.6f}    {status}")

    # Overall result
    final_success = results['all'][0]
    print(f"\n{'='*80}")
    if final_success:
        print(f"{'🎉 ALL TARGETS ACHIEVED! 🎉':^80}")
    else:
        print(f"{'⚠ Training Complete - Some targets not fully met':^80}")
    print(f"{'='*80}\n")

    print(f"Saved checkpoints:")
    print(f"  • checkpoints/model_clean_95_DIV2K.pth")
    print(f"  • checkpoints/model_jpeg_95_DIV2K.pth")
    print(f"  • checkpoints/model_blur_resize_95_DIV2K.pth")
    print(f"  • checkpoints/model_ALL_TRANSFORMATIONS_95_DIV2K.pth (FINAL)\n")


if __name__ == "__main__":
    main()
