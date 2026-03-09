"""
Example training script demonstrating the use of new distortions:
- Gaussian Blur
- Resize Attack  
- Color Jitter

This script shows how to train the model with enhanced robustness.
"""

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from models.model import StegoModel
import os


class DummyDataset(Dataset):
    """Dummy dataset for demonstration."""

    def __init__(self, num_samples=100, image_size=256, message_length=1024):
        self.num_samples = num_samples
        self.image_size = image_size
        self.message_length = message_length

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random cover image and binary message
        cover = torch.rand(3, self.image_size, self.image_size)
        message = torch.randint(0, 2, (self.message_length,)).float()
        return cover, message


def train_with_new_distortions():
    """Train model with Gaussian blur, resize attack, and color jitter."""

    print("=" * 70)
    print("Training Steganography Model with Enhanced Distortions")
    print("=" * 70)
    print("\nNew distortions enabled:")
    print("  ✓ Gaussian Blur (30% probability)")
    print("  ✓ Resize Attack (30% probability)")
    print("  ✓ Color Jitter (30% probability)")
    print()

    # Hyperparameters (matching standard training configuration)
    message_length = 1024
    image_size = 128  # Standard size from config.yaml
    batch_size = 4    # Standard batch size from config.yaml
    num_epochs = 5
    learning_rate = 1e-4
    alpha = 1.0  # Image loss weight
    beta = 1.0   # Message loss weight

    # Create model with distortions enabled
    model = StegoModel(
        message_length=message_length,
        image_size=image_size,
        enable_distortions=True  # This enables all distortions including the new ones
    )

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Training on: {device}\n")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    # Dataset and DataLoader
    train_dataset = DummyDataset(
        num_samples=100, image_size=image_size, message_length=message_length)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    print("=" * 70)
    print("Starting Training...")
    print("=" * 70)

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_image_loss = 0.0
        epoch_message_loss = 0.0
        epoch_accuracy = 0.0
        num_batches = 0

        for batch_idx, (cover_images, binary_messages) in enumerate(train_loader):
            # Move to device
            cover_images = cover_images.to(device)
            binary_messages = binary_messages.to(device)

            # Forward pass with distortions (including new ones)
            optimizer.zero_grad()
            loss_dict = model.compute_loss(
                cover_images,
                binary_messages,
                alpha=alpha,
                beta=beta
            )

            # Backward pass
            loss_dict['total_loss'].backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Accumulate metrics
            epoch_loss += loss_dict['total_loss'].item()
            epoch_image_loss += loss_dict['image_loss'].item()
            epoch_message_loss += loss_dict['message_loss'].item()
            epoch_accuracy += loss_dict['bit_accuracy'].item()
            num_batches += 1

            # Print progress
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {loss_dict['total_loss'].item():.4f} "
                      f"Acc: {loss_dict['bit_accuracy'].item()*100:.2f}%")

        # Epoch statistics
        avg_loss = epoch_loss / num_batches
        avg_image_loss = epoch_image_loss / num_batches
        avg_message_loss = epoch_message_loss / num_batches
        avg_accuracy = epoch_accuracy / num_batches

        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{num_epochs} Summary:")
        print(f"{'='*70}")
        print(f"  Average Total Loss:   {avg_loss:.6f}")
        print(f"  Average Image Loss:   {avg_image_loss:.6f}")
        print(f"  Average Message Loss: {avg_message_loss:.6f}")
        print(f"  Average Bit Accuracy: {avg_accuracy*100:.2f}%")
        print(f"{'='*70}\n")

        # Update learning rate
        scheduler.step(avg_loss)

    print("=" * 70)
    print("Training Complete!")
    print("=" * 70)

    # Save model
    save_path = 'checkpoints/model_with_new_distortions.pth'
    os.makedirs('checkpoints', exist_ok=True)
    model.save_model(save_path)
    print(f"\nModel saved to: {save_path}")

    # Test robustness
    print("\n" + "=" * 70)
    print("Testing Robustness to New Attacks...")
    print("=" * 70)

    model.eval()
    test_cover = torch.rand(1, 3, image_size, image_size).to(device)
    test_message = torch.randint(0, 2, (1, message_length)).float().to(device)

    with torch.no_grad():
        # Clean encoding/decoding
        stego = model.encode(test_cover, test_message)
        decoded_clean = model.decode(stego)
        accuracy_clean = (decoded_clean == test_message).float().mean()

        # Test with individual attacks
        distortions = model.distortions

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

    print(f"\nRobustness Results:")
    print(f"  Clean (no attack):        {accuracy_clean.item()*100:.2f}%")
    print(f"  After Gaussian Blur:      {accuracy_blur.item()*100:.2f}%")
    print(f"  After Resize Attack:      {accuracy_resize.item()*100:.2f}%")
    print(f"  After Color Jitter:       {accuracy_jitter.item()*100:.2f}%")

    print("\n" + "=" * 70)
    print("Done! The model has been trained with enhanced robustness.")
    print("=" * 70)


if __name__ == "__main__":
    train_with_new_distortions()
