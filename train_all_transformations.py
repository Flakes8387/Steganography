"""
Progressive Training for All Three Transformations
Target: 80%+ accuracy AND pixel delta ≤ 0.02
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models.model import StegoModel
import os
from datetime import datetime


def calculate_psnr(img1, img2):
    """Calculate PSNR between two images."""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()


def evaluate_model(model, dataloader, device, phase_name=""):
    """Evaluate model and return detailed metrics."""
    model.eval()
    total_accuracy = 0
    total_pixel_delta = 0
    total_psnr = 0
    total_ber = 0
    num_batches = 0

    with torch.no_grad():
        for cover_images, binary_messages in dataloader:
            cover_images = cover_images.to(device)
            binary_messages = binary_messages.to(device)

            # Forward pass
            outputs = model(cover_images, binary_messages,
                            apply_distortions=False)

            # Calculate metrics
            decoded = outputs['decoded_message']
            stego = outputs['stego_image']

            accuracy = (decoded == binary_messages).float().mean()
            pixel_delta = torch.mean(torch.abs(stego - cover_images))
            psnr = calculate_psnr(cover_images, stego)
            ber = 1 - accuracy

            total_accuracy += accuracy.item()
            total_pixel_delta += pixel_delta.item()
            total_psnr += psnr
            total_ber += ber.item()
            num_batches += 1

    avg_accuracy = total_accuracy / num_batches
    avg_pixel_delta = total_pixel_delta / num_batches
    avg_psnr = total_psnr / num_batches
    avg_ber = total_ber / num_batches

    print(f"\n{'='*70}")
    print(f"{phase_name} EVALUATION RESULTS:")
    print(f"{'='*70}")
    print(f"  Accuracy:    {avg_accuracy*100:.2f}%")
    print(f"  Pixel Delta: {avg_pixel_delta:.6f}")
    print(f"  PSNR:        {avg_psnr:.2f} dB")
    print(f"  BER:         {avg_ber*100:.2f}%")

    # Check if targets are met
    target_met = avg_accuracy >= 0.80 and avg_pixel_delta <= 0.02
    status = "✓ TARGET MET" if target_met else "✗ TARGET NOT MET"
    print(f"  Status:      {status}")
    print(f"{'='*70}\n")

    model.train()
    return avg_accuracy, avg_pixel_delta, target_met


def train_phase(model, dataloader, optimizer, scheduler, device, phase_name,
                max_epochs=20, target_accuracy=0.80, target_pixel_delta=0.02):
    """Train for one phase until targets are met."""

    print(f"\n{'#'*70}")
    print(f"# {phase_name}")
    print(
        f"# Target: Accuracy ≥ {target_accuracy*100}% AND Pixel Delta ≤ {target_pixel_delta}")
    print(f"{'#'*70}\n")

    best_accuracy = 0
    best_pixel_delta = float('inf')
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        epoch_accuracy = 0
        epoch_pixel_delta = 0
        epoch_ber = 0
        epoch_psnr = 0
        num_batches = 0

        print(f"\nEpoch {epoch+1}/{max_epochs}")
        print("-" * 70)

        for batch_idx, (cover_images, binary_messages) in enumerate(dataloader):
            cover_images = cover_images.to(device)
            binary_messages = binary_messages.to(device)

            # Forward pass
            optimizer.zero_grad()
            loss_dict = model.compute_loss(
                cover_images, binary_messages, alpha=1.0, beta=1.5)

            # Backward pass
            loss_dict['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Calculate metrics
            with torch.no_grad():
                outputs = model(cover_images, binary_messages,
                                apply_distortions=False)
                stego = outputs['stego_image']
                pixel_delta = torch.mean(torch.abs(stego - cover_images))
                psnr = calculate_psnr(cover_images, stego)

            accuracy = loss_dict['bit_accuracy'].item()
            ber = 1 - accuracy

            epoch_loss += loss_dict['total_loss'].item()
            epoch_accuracy += accuracy
            epoch_pixel_delta += pixel_delta.item()
            epoch_ber += ber
            epoch_psnr += psnr
            num_batches += 1

            # Print batch details every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx+1:3d} | "
                      f"Loss: {loss_dict['total_loss'].item():.4f} | "
                      f"Acc: {accuracy*100:5.2f}% | "
                      f"BER: {ber*100:5.2f}% | "
                      f"Delta: {pixel_delta.item():.6f} | "
                      f"PSNR: {psnr:5.2f} dB")

        # Epoch summary
        avg_loss = epoch_loss / num_batches
        avg_accuracy = epoch_accuracy / num_batches
        avg_pixel_delta = epoch_pixel_delta / num_batches
        avg_ber = epoch_ber / num_batches
        avg_psnr = epoch_psnr / num_batches

        print(f"\n  EPOCH SUMMARY:")
        print(f"  Loss: {avg_loss:.4f} | Acc: {avg_accuracy*100:.2f}% | "
              f"Delta: {avg_pixel_delta:.6f} | PSNR: {avg_psnr:.2f} dB")

        # Update learning rate
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Learning Rate: {current_lr:.2e}")

        # Check if targets are met
        target_met = avg_accuracy >= target_accuracy and avg_pixel_delta <= target_pixel_delta

        # Track best performance
        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if avg_pixel_delta < best_pixel_delta:
            best_pixel_delta = avg_pixel_delta

        print(
            f"  Best Accuracy: {best_accuracy*100:.2f}% | Best Delta: {best_pixel_delta:.6f}")

        if target_met:
            print(
                f"\n  ✓ TARGET ACHIEVED! Accuracy: {avg_accuracy*100:.2f}%, Delta: {avg_pixel_delta:.6f}")
            return True, avg_accuracy, avg_pixel_delta

        # Early stopping if no improvement
        if epochs_without_improvement >= 5:
            print(f"\n  ⚠ No improvement for 5 epochs. Stopping early.")
            break

    print(f"\n  ✗ Target not achieved after {max_epochs} epochs.")
    print(
        f"  Final: Accuracy: {avg_accuracy*100:.2f}%, Delta: {avg_pixel_delta:.6f}")
    return False, avg_accuracy, avg_pixel_delta


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Configuration
    message_length = 16
    image_size = 128
    batch_size = 4
    num_samples = 200
    max_epochs_per_phase = 20

    print(f"\nConfiguration:")
    print(f"  Message Length: {message_length}")
    print(f"  Image Size: {image_size}x{image_size}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Samples: {num_samples}")

    # Generate synthetic dataset
    print(f"\nGenerating synthetic dataset...")
    cover_images = torch.rand(num_samples, 3, image_size, image_size)
    binary_messages = torch.randint(
        0, 2, (num_samples, message_length)).float()
    dataset = TensorDataset(cover_images, binary_messages)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"  ✓ Dataset ready: {num_samples} samples")

    # Phase 1: Train with Gaussian Blur
    print(f"\n{'#'*70}")
    print(f"# PHASE 1: GAUSSIAN BLUR")
    print(f"{'#'*70}")

    # Load checkpoint with blur
    if os.path.exists('checkpoints/model_with_blur.pth'):
        print(f"\nLoading checkpoint: checkpoints/model_with_blur.pth")
        model = StegoModel(message_length, image_size, enable_distortions=True)
        checkpoint = torch.load(
            'checkpoints/model_with_blur.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        print(f"  ✓ Checkpoint loaded")
    else:
        print(f"\n⚠ Checkpoint not found. Loading base model...")
        model = StegoModel(message_length, image_size, enable_distortions=True)
        checkpoint = torch.load(
            'checkpoints/best_model_local.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

    # Baseline evaluation
    accuracy_baseline, delta_baseline, _ = evaluate_model(
        model, dataloader, device, "BASELINE")

    # Configure distortions for Phase 1: Gaussian Blur only
    def forward_blur_only(images, apply_all=False, jpeg_only=False):
        if model.distortions.training:
            images = model.distortions.apply_gaussian_blur_attack(
                images, probability=0.5)
        return images

    original_forward = model.distortions.forward
    model.distortions.forward = forward_blur_only

    # Train Phase 1
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3)

    success_blur, acc_blur, delta_blur = train_phase(
        model, dataloader, optimizer, scheduler, device,
        "PHASE 1: GAUSSIAN BLUR TRAINING",
        max_epochs=max_epochs_per_phase
    )

    # Save Phase 1
    torch.save({
        'model_state_dict': model.state_dict(),
        'message_length': message_length,
        'image_size': image_size,
        'accuracy': acc_blur,
        'pixel_delta': delta_blur,
        'phase': 'gaussian_blur'
    }, 'checkpoints/model_with_blur_final.pth')
    print(f"\n✓ Phase 1 model saved: checkpoints/model_with_blur_final.pth")

    # Phase 2: Add Resize Attack
    print(f"\n{'#'*70}")
    print(f"# PHASE 2: RESIZE ATTACK (Building on Blur)")
    print(f"{'#'*70}")

    # Configure distortions for Phase 2: Blur + Resize
    def forward_blur_resize(images, apply_all=False, jpeg_only=False):
        if model.distortions.training:
            images = model.distortions.apply_gaussian_blur_attack(
                images, probability=0.4)
            images = model.distortions.apply_resize_attack(
                images, probability=0.4)
        return images

    model.distortions.forward = forward_blur_resize

    # Continue with same model
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3)

    success_resize, acc_resize, delta_resize = train_phase(
        model, dataloader, optimizer, scheduler, device,
        "PHASE 2: RESIZE ATTACK TRAINING",
        max_epochs=max_epochs_per_phase
    )

    # Save Phase 2
    torch.save({
        'model_state_dict': model.state_dict(),
        'message_length': message_length,
        'image_size': image_size,
        'accuracy': acc_resize,
        'pixel_delta': delta_resize,
        'phase': 'blur_resize'
    }, 'checkpoints/model_with_blur_resize_final.pth')
    print(f"\n✓ Phase 2 model saved: checkpoints/model_with_blur_resize_final.pth")

    # Phase 3: Add Color Jitter
    print(f"\n{'#'*70}")
    print(f"# PHASE 3: COLOR JITTER (Building on Blur + Resize)")
    print(f"{'#'*70}")

    # Configure distortions for Phase 3: All three
    def forward_all(images, apply_all=False, jpeg_only=False):
        if model.distortions.training:
            images = model.distortions.apply_gaussian_blur_attack(
                images, probability=0.3)
            images = model.distortions.apply_resize_attack(
                images, probability=0.3)
            images = model.distortions.apply_color_jitter_attack(
                images, probability=0.3)
        return images

    model.distortions.forward = forward_all

    # Continue with same model
    optimizer = optim.Adam(model.parameters(), lr=3e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3)

    success_jitter, acc_jitter, delta_jitter = train_phase(
        model, dataloader, optimizer, scheduler, device,
        "PHASE 3: COLOR JITTER TRAINING",
        max_epochs=max_epochs_per_phase
    )

    # Save Final Model
    torch.save({
        'model_state_dict': model.state_dict(),
        'message_length': message_length,
        'image_size': image_size,
        'accuracy': acc_jitter,
        'pixel_delta': delta_jitter,
        'phase': 'all_transformations'
    }, 'checkpoints/model_all_transformations_final.pth')
    print(f"\n✓ Final model saved: checkpoints/model_all_transformations_final.pth")

    # Final Summary
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE - FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"\nBaseline:")
    print(
        f"  Accuracy: {accuracy_baseline*100:.2f}% | Pixel Delta: {delta_baseline:.6f}")
    print(f"\nPhase 1 (Gaussian Blur):")
    print(
        f"  Accuracy: {acc_blur*100:.2f}% | Pixel Delta: {delta_blur:.6f} | {'✓ SUCCESS' if success_blur else '✗ FAILED'}")
    print(f"\nPhase 2 (+ Resize Attack):")
    print(
        f"  Accuracy: {acc_resize*100:.2f}% | Pixel Delta: {delta_resize:.6f} | {'✓ SUCCESS' if success_resize else '✗ FAILED'}")
    print(f"\nPhase 3 (+ Color Jitter):")
    print(
        f"  Accuracy: {acc_jitter*100:.2f}% | Pixel Delta: {delta_jitter:.6f} | {'✓ SUCCESS' if success_jitter else '✗ FAILED'}")

    # Overall status
    overall_success = acc_jitter >= 0.80 and delta_jitter <= 0.02
    print(
        f"\nOVERALL STATUS: {'✓ ALL TARGETS MET' if overall_success else '✗ TARGETS NOT MET'}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
