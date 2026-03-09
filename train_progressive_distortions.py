"""
Progressive Distortion Training Script

Trains the model by adding distortions one at a time:
1. Start with clean training (or load existing best model)
2. Add Gaussian Blur
3. Add Resize Attack
4. Add Color Jitter

Target: Achieve 80%+ bit accuracy with all distortions.

Features:
- Loads existing best model without altering original
- Saves progressive checkpoints
- Shows detailed metrics (loss, accuracy, pixel delta, PSNR)
- Stops when 80% accuracy is reached
"""

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from models.model import StegoModel
import os
from datetime import datetime
import copy


class DummyDataset(Dataset):
    """Dataset for demonstration."""

    def __init__(self, num_samples=200, image_size=128, message_length=1024):
        self.num_samples = num_samples
        self.image_size = image_size
        self.message_length = message_length

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        cover = torch.rand(3, self.image_size, self.image_size)
        message = torch.randint(0, 2, (self.message_length,)).float()
        return cover, message


def calculate_psnr(img1, img2):
    """Calculate Peak Signal-to-Noise Ratio."""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()


def evaluate_model(model, dataloader, device, phase_name=""):
    """Evaluate model and return detailed metrics."""
    model.eval()
    total_loss = 0.0
    total_image_loss = 0.0
    total_message_loss = 0.0
    total_accuracy = 0.0
    total_psnr = 0.0
    total_pixel_delta = 0.0
    num_batches = 0

    with torch.no_grad():
        for cover_images, binary_messages in dataloader:
            cover_images = cover_images.to(device)
            binary_messages = binary_messages.to(device)

            # Get outputs
            outputs = model(cover_images, binary_messages)
            loss_dict = model.compute_loss(cover_images, binary_messages)

            # Calculate metrics
            stego = outputs['stego_image']
            pixel_delta = torch.abs(stego - cover_images).mean()
            psnr = calculate_psnr(cover_images, stego)

            total_loss += loss_dict['total_loss'].item()
            total_image_loss += loss_dict['image_loss'].item()
            total_message_loss += loss_dict['message_loss'].item()
            total_accuracy += loss_dict['bit_accuracy'].item()
            total_psnr += psnr
            total_pixel_delta += pixel_delta.item()
            num_batches += 1

    metrics = {
        'loss': total_loss / num_batches,
        'image_loss': total_image_loss / num_batches,
        'message_loss': total_message_loss / num_batches,
        'accuracy': total_accuracy / num_batches,
        'psnr': total_psnr / num_batches,
        'pixel_delta': total_pixel_delta / num_batches
    }

    model.train()
    return metrics


def train_phase(model, train_loader, val_loader, optimizer, device,
                phase_name, target_accuracy=0.80, max_epochs=50):
    """Train for one phase until target accuracy is reached."""

    print(f"\n{'='*80}")
    print(f"PHASE: {phase_name}")
    print(f"{'='*80}")
    print(f"Target Accuracy: {target_accuracy*100:.1f}%")
    print(f"Max Epochs: {max_epochs}")
    print()

    best_accuracy = 0.0
    epochs_without_improvement = 0
    patience = 10

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        epoch_pixel_delta = 0.0
        num_batches = 0

        print(f"\nEpoch [{epoch+1}/{max_epochs}]")
        print("-" * 80)

        for batch_idx, (cover_images, binary_messages) in enumerate(train_loader):
            cover_images = cover_images.to(device)
            binary_messages = binary_messages.to(device)

            # Forward pass
            optimizer.zero_grad()

            # Get stego image for pixel delta calculation
            with torch.no_grad():
                stego = model.encode(cover_images, binary_messages)
                pixel_delta = torch.abs(stego - cover_images).mean()

            # Compute loss
            loss_dict = model.compute_loss(
                cover_images, binary_messages, alpha=1.0, beta=1.0)

            # Backward pass
            loss_dict['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Accumulate metrics
            epoch_loss += loss_dict['total_loss'].item()
            epoch_accuracy += loss_dict['bit_accuracy'].item()
            epoch_pixel_delta += pixel_delta.item()
            num_batches += 1

            # Print batch progress
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"  Batch [{batch_idx+1:3d}/{len(train_loader)}] | "
                      f"Loss: {loss_dict['total_loss'].item():.4f} | "
                      f"Acc: {loss_dict['bit_accuracy'].item()*100:5.2f}% | "
                      f"BER: {loss_dict['ber'].item()*100:5.2f}% | "
                      f"Pixel Δ: {pixel_delta.item():.6f}")

        # Epoch statistics
        avg_loss = epoch_loss / num_batches
        avg_accuracy = epoch_accuracy / num_batches
        avg_pixel_delta = epoch_pixel_delta / num_batches

        # Validation
        val_metrics = evaluate_model(model, val_loader, device, phase_name)

        print(f"\n  {'TRAINING METRICS':^80}")
        print(f"  {'-'*80}")
        print(f"  Loss:        {avg_loss:.6f}")
        print(f"  Accuracy:    {avg_accuracy*100:5.2f}%")
        print(f"  Pixel Delta: {avg_pixel_delta:.6f}")

        print(f"\n  {'VALIDATION METRICS':^80}")
        print(f"  {'-'*80}")
        print(f"  Loss:          {val_metrics['loss']:.6f}")
        print(f"  Image Loss:    {val_metrics['image_loss']:.6f}")
        print(f"  Message Loss:  {val_metrics['message_loss']:.6f}")
        print(f"  Accuracy:      {val_metrics['accuracy']*100:5.2f}%")
        print(f"  BER:           {(1-val_metrics['accuracy'])*100:5.2f}%")
        print(f"  PSNR:          {val_metrics['psnr']:.2f} dB")
        print(f"  Pixel Delta:   {val_metrics['pixel_delta']:.6f}")

        # Check if target reached
        if val_metrics['accuracy'] >= target_accuracy:
            print(
                f"\n  🎉 TARGET REACHED! Accuracy: {val_metrics['accuracy']*100:.2f}% >= {target_accuracy*100:.1f}%")
            return val_metrics['accuracy'], epoch + 1

        # Early stopping check
        if val_metrics['accuracy'] > best_accuracy:
            best_accuracy = val_metrics['accuracy']
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(
                    f"\n  ⚠ Early stopping: No improvement for {patience} epochs")
                print(f"  Best accuracy achieved: {best_accuracy*100:.2f}%")
                return best_accuracy, epoch + 1

    print(f"\n  ⚠ Max epochs reached. Best accuracy: {best_accuracy*100:.2f}%")
    return best_accuracy, max_epochs


def progressive_distortion_training():
    """Main training function with progressive distortion integration."""

    print("=" * 80)
    print("PROGRESSIVE DISTORTION TRAINING")
    print("=" * 80)
    print("\nGoal: Train model to 80%+ accuracy with Gaussian blur, resize, and color jitter")
    print("\nTraining Strategy:")
    print("  Phase 1: Load existing best model (baseline)")
    print("  Phase 2: Add Gaussian Blur → Train to 80%")
    print("  Phase 3: Add Resize Attack → Train to 80%")
    print("  Phase 4: Add Color Jitter → Train to 80%")
    print()

    # Configuration (matching standard training)
    # Note: Will auto-detect message_length from checkpoint if available
    message_length = 1024  # Default, will be overridden if checkpoint exists
    image_size = 128
    batch_size = 4
    learning_rate = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Check if checkpoint exists and detect message_length
    checkpoint_path = 'checkpoints/best_model_local.pth'
    if os.path.exists(checkpoint_path):
        temp_checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in temp_checkpoint:
            # Detect message length from fc3.bias shape
            fc3_bias_shape = temp_checkpoint['model_state_dict']['decoder.message_extractor.fc3.bias'].shape[0]
            message_length = fc3_bias_shape
            print(
                f"\n✓ Detected message_length from checkpoint: {message_length}")
        del temp_checkpoint

    print(f"Configuration:")
    print(f"  Device: {device}")
    print(f"  Image size: {image_size}x{image_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Message length: {message_length} bits")
    print(f"  Learning rate: {learning_rate}")

    # Create datasets
    train_dataset = DummyDataset(
        num_samples=200, image_size=image_size, message_length=message_length)
    val_dataset = DummyDataset(
        num_samples=50, image_size=image_size, message_length=message_length)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"\nDataset:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")

    # ========================================================================
    # PHASE 1: Load existing best model
    # ========================================================================

    if os.path.exists(checkpoint_path):
        print(f"\n{'='*80}")
        print(f"PHASE 1: LOADING EXISTING BEST MODEL")
        print(f"{'='*80}")
        print(f"Loading from: {checkpoint_path}")

        # Create model and load checkpoint (without altering original)
        model = StegoModel(
            message_length=message_length,
            image_size=image_size,
            enable_distortions=False  # Start clean
        ).to(device)

        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load model state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'encoder_state_dict' in checkpoint and 'decoder_state_dict' in checkpoint:
            model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            model.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        else:
            print("⚠ Unknown checkpoint format, starting from scratch")
            checkpoint = None

        if checkpoint:
            print("✓ Model loaded successfully")
            if 'epoch' in checkpoint:
                print(f"  Checkpoint from epoch: {checkpoint['epoch']}")
            if 'metrics' in checkpoint:
                print(f"  Checkpoint metrics: {checkpoint['metrics']}")

        # Evaluate baseline
        print("\nEvaluating baseline performance...")
        baseline_metrics = evaluate_model(
            model, val_loader, device, "Baseline")

        print(f"\n  {'BASELINE METRICS (Clean, No Distortions)':^80}")
        print(f"  {'-'*80}")
        print(f"  Accuracy:    {baseline_metrics['accuracy']*100:5.2f}%")
        print(f"  PSNR:        {baseline_metrics['psnr']:.2f} dB")
        print(f"  Pixel Delta: {baseline_metrics['pixel_delta']:.6f}")

        # Save backup of original model
        backup_path = checkpoint_path.replace('.pth', '_original_backup.pth')
        if not os.path.exists(backup_path):
            torch.save(checkpoint, backup_path)
            print(f"\n✓ Original model backed up to: {backup_path}")
    else:
        print(f"\n{'='*80}")
        print(f"PHASE 1: CREATING NEW MODEL")
        print(f"{'='*80}")
        print(f"No existing checkpoint found at: {checkpoint_path}")
        print("Creating new model from scratch...")

        model = StegoModel(
            message_length=message_length,
            image_size=image_size,
            enable_distortions=False
        ).to(device)

        baseline_metrics = {'accuracy': 0.0}

    # ========================================================================
    # PHASE 2: Add Gaussian Blur
    # ========================================================================
    print(f"\n{'='*80}")
    print(f"PHASE 2: TRAINING WITH GAUSSIAN BLUR")
    print(f"{'='*80}")

    # Enable Gaussian blur only
    model.distortions.train()
    original_forward = model.distortions.forward

    def forward_with_blur_only(images, apply_all=False, jpeg_only=False):
        """Apply only Gaussian blur."""
        if model.distortions.training:
            images = model.distortions.apply_gaussian_blur_attack(
                images, probability=0.5)
        return images

    model.distortions.forward = forward_with_blur_only
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    blur_accuracy, blur_epochs = train_phase(
        model, train_loader, val_loader, optimizer, device,
        "Gaussian Blur Only", target_accuracy=0.80, max_epochs=30
    )

    # Save checkpoint
    save_path = 'checkpoints/model_with_gaussian_blur.pth'
    model.save_model(save_path)
    print(f"\n✓ Model saved: {save_path}")
    print(
        f"  Final accuracy: {blur_accuracy*100:.2f}% (trained for {blur_epochs} epochs)")

    # ========================================================================
    # PHASE 3: Add Resize Attack
    # ========================================================================
    print(f"\n{'='*80}")
    print(f"PHASE 3: TRAINING WITH GAUSSIAN BLUR + RESIZE ATTACK")
    print(f"{'='*80}")

    def forward_with_blur_and_resize(images, apply_all=False, jpeg_only=False):
        """Apply Gaussian blur and resize attack."""
        if model.distortions.training:
            images = model.distortions.apply_gaussian_blur_attack(
                images, probability=0.4)
            images = model.distortions.apply_resize_attack(
                images, probability=0.4)
        return images

    model.distortions.forward = forward_with_blur_and_resize
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate * 0.5)  # Lower LR

    resize_accuracy, resize_epochs = train_phase(
        model, train_loader, val_loader, optimizer, device,
        "Gaussian Blur + Resize Attack", target_accuracy=0.80, max_epochs=30
    )

    # Save checkpoint
    save_path = 'checkpoints/model_with_blur_resize.pth'
    model.save_model(save_path)
    print(f"\n✓ Model saved: {save_path}")
    print(
        f"  Final accuracy: {resize_accuracy*100:.2f}% (trained for {resize_epochs} epochs)")

    # ========================================================================
    # PHASE 4: Add Color Jitter
    # ========================================================================
    print(f"\n{'='*80}")
    print(f"PHASE 4: TRAINING WITH ALL DISTORTIONS")
    print(f"{'='*80}")
    print("Adding: Gaussian Blur + Resize Attack + Color Jitter")

    def forward_with_all_distortions(images, apply_all=False, jpeg_only=False):
        """Apply all three distortions."""
        if model.distortions.training:
            images = model.distortions.apply_gaussian_blur_attack(
                images, probability=0.3)
            images = model.distortions.apply_resize_attack(
                images, probability=0.3)
            images = model.distortions.apply_color_jitter_attack(
                images, probability=0.3)
        return images

    model.distortions.forward = forward_with_all_distortions
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate * 0.3)  # Even lower LR

    final_accuracy, final_epochs = train_phase(
        model, train_loader, val_loader, optimizer, device,
        "All Distortions (Blur + Resize + Color Jitter)", target_accuracy=0.80, max_epochs=40
    )

    # Save final checkpoint
    save_path = 'checkpoints/model_with_all_new_distortions.pth'
    model.save_model(save_path)
    print(f"\n✓ Model saved: {save_path}")
    print(
        f"  Final accuracy: {final_accuracy*100:.2f}% (trained for {final_epochs} epochs)")

    # ========================================================================
    # FINAL EVALUATION
    # ========================================================================
    print(f"\n{'='*80}")
    print(f"FINAL EVALUATION - ROBUSTNESS TEST")
    print(f"{'='*80}")

    # Restore original forward method for comprehensive testing
    model.distortions.forward = original_forward
    model.eval()

    test_cover = torch.rand(1, 3, image_size, image_size).to(device)
    test_message = torch.randint(0, 2, (1, message_length)).float().to(device)

    with torch.no_grad():
        # Clean encoding
        stego = model.encode(test_cover, test_message)
        decoded_clean = model.decode(stego)
        accuracy_clean = (decoded_clean == test_message).float().mean()
        psnr_clean = calculate_psnr(test_cover, stego)
        pixel_delta_clean = torch.abs(stego - test_cover).mean()

        # Test individual attacks
        distortions = model.distortions
        distortions.train()

        # Gaussian blur
        blurred = distortions.apply_gaussian_blur_attack(
            stego, probability=1.0)
        decoded_blur = model.decode(blurred)
        accuracy_blur = (decoded_blur == test_message).float().mean()

        # Resize attack
        resized = distortions.apply_resize_attack(stego, probability=1.0)
        decoded_resize = model.decode(resized)
        accuracy_resize = (decoded_resize == test_message).float().mean()

        # Color jitter
        jittered = distortions.apply_color_jitter_attack(
            stego, probability=1.0)
        decoded_jitter = model.decode(jittered)
        accuracy_jitter = (decoded_jitter == test_message).float().mean()

        # All combined
        combined = distortions.apply_gaussian_blur_attack(
            stego, probability=1.0)
        combined = distortions.apply_resize_attack(combined, probability=1.0)
        combined = distortions.apply_color_jitter_attack(
            combined, probability=1.0)
        decoded_combined = model.decode(combined)
        accuracy_combined = (decoded_combined == test_message).float().mean()

    print(f"\nRobustness Results:")
    print(f"  {'Attack Type':<30} {'Accuracy':<12} {'Status'}")
    print(f"  {'-'*80}")
    print(f"  {'Clean (no attack)':<30} {accuracy_clean.item()*100:>5.2f}%     "
          f"  {'✓ PASS' if accuracy_clean.item() >= 0.80 else '✗ FAIL'}")
    print(f"  {'Gaussian Blur':<30} {accuracy_blur.item()*100:>5.2f}%     "
          f"  {'✓ PASS' if accuracy_blur.item() >= 0.80 else '✗ FAIL'}")
    print(f"  {'Resize Attack':<30} {accuracy_resize.item()*100:>5.2f}%     "
          f"  {'✓ PASS' if accuracy_resize.item() >= 0.80 else '✗ FAIL'}")
    print(f"  {'Color Jitter':<30} {accuracy_jitter.item()*100:>5.2f}%     "
          f"  {'✓ PASS' if accuracy_jitter.item() >= 0.80 else '✗ FAIL'}")
    print(f"  {'All Combined':<30} {accuracy_combined.item()*100:>5.2f}%     "
          f"  {'✓ PASS' if accuracy_combined.item() >= 0.80 else '✗ FAIL'}")

    print(f"\nImage Quality Metrics:")
    print(f"  PSNR:        {psnr_clean:.2f} dB")
    print(f"  Pixel Delta: {pixel_delta_clean.item():.6f}")

    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE!")
    print(f"{'='*80}")

    # Summary
    print(f"\nTraining Summary:")
    print(
        f"  Phase 1 - Baseline:      {baseline_metrics['accuracy']*100:5.2f}%")
    print(
        f"  Phase 2 - + Blur:        {blur_accuracy*100:5.2f}% ({blur_epochs} epochs)")
    print(
        f"  Phase 3 - + Resize:      {resize_accuracy*100:5.2f}% ({resize_epochs} epochs)")
    print(
        f"  Phase 4 - + Color Jitter: {final_accuracy*100:5.2f}% ({final_epochs} epochs)")

    if final_accuracy >= 0.80:
        print(
            f"\n🎉 SUCCESS! Final accuracy {final_accuracy*100:.2f}% exceeds 80% target!")
    else:
        print(f"\n⚠ Target not fully reached. Continue training or adjust hyperparameters.")

    print(f"\nSaved Checkpoints:")
    print(f"  - checkpoints/best_model_local_original_backup.pth (original)")
    print(f"  - checkpoints/model_with_gaussian_blur.pth")
    print(f"  - checkpoints/model_with_blur_resize.pth")
    print(f"  - checkpoints/model_with_all_new_distortions.pth (final)")


if __name__ == "__main__":
    progressive_distortion_training()
