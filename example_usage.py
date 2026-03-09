"""
Example usage of the unified StegoModel for training and inference.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from models.model import StegoModel


def example_training():
    """Example of training the steganography model."""

    print("=" * 70)
    print("EXAMPLE: Training Steganography Model")
    print("=" * 70)

    # Configuration
    message_length = 1024
    image_size = 256
    batch_size = 4
    num_epochs = 3
    learning_rate = 1e-4

    # Create model
    print("\n1. Creating model...")
    model = StegoModel(
        message_length=message_length,
        image_size=image_size,
        enable_distortions=True
    )

    params = model.get_num_parameters()
    print(f"   Total parameters: {params['total']:,}")

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Simulated training loop
    print(f"\n2. Training for {num_epochs} epochs...")
    model.train()

    for epoch in range(num_epochs):
        # Generate random batch (in real scenario, load from dataset)
        cover_images = torch.rand(batch_size, 3, image_size, image_size)
        binary_messages = torch.randint(
            0, 2, (batch_size, message_length)).float()

        # Forward pass and compute loss
        optimizer.zero_grad()
        loss_dict = model.compute_loss(
            cover_images,
            binary_messages,
            alpha=1.0,  # Weight for image loss
            beta=1.0    # Weight for message loss
        )

        # Backward pass
        loss_dict['total_loss'].backward()
        optimizer.step()

        # Print progress
        print(f"   Epoch {epoch+1}/{num_epochs}")
        print(f"      Loss: {loss_dict['total_loss'].item():.6f}")
        print(f"      Image Loss: {loss_dict['image_loss'].item():.6f}")
        print(f"      Message Loss: {loss_dict['message_loss'].item():.6f}")
        print(f"      Accuracy: {loss_dict['accuracy'].item()*100:.2f}%")

    print("\n3. Training complete!")
    return model


def example_inference():
    """Example of using the model for inference."""

    print("\n" + "=" * 70)
    print("EXAMPLE: Inference with Steganography Model")
    print("=" * 70)

    # Configuration
    message_length = 1024
    image_size = 256

    # Create model
    print("\n1. Creating model...")
    model = StegoModel(
        message_length=message_length,
        image_size=image_size,
        enable_distortions=False  # No distortions for inference
    )
    model.eval()

    # Create sample data
    print("\n2. Preparing data...")
    cover_image = torch.rand(1, 3, image_size, image_size)
    binary_message = torch.randint(0, 2, (1, message_length)).float()

    print(f"   Cover image shape: {cover_image.shape}")
    print(f"   Binary message: {binary_message.shape}")
    print(f"   Message (first 20 bits): {binary_message[0, :20].tolist()}")

    # Encode
    print("\n3. Encoding message into image...")
    with torch.no_grad():
        stego_image = model.encode(cover_image, binary_message)

    print(f"   Stego image shape: {stego_image.shape}")

    # Calculate image difference
    diff = torch.abs(stego_image - cover_image)
    print(f"   Average pixel difference: {diff.mean().item():.6f}")
    print(f"   Max pixel difference: {diff.max().item():.6f}")
    print(f"   PSNR: {calculate_psnr(cover_image, stego_image):.2f} dB")

    # Decode
    print("\n4. Decoding message from stego image...")
    with torch.no_grad():
        decoded_message = model.decode(stego_image)

    print(f"   Decoded message shape: {decoded_message.shape}")
    print(f"   Decoded (first 20 bits): {decoded_message[0, :20].tolist()}")

    # Calculate accuracy
    accuracy = (decoded_message == binary_message).float().mean()
    print(f"   Bit accuracy: {accuracy.item()*100:.2f}%")

    # Test with distortions
    print("\n5. Testing robustness with distortions...")
    model_with_dist = StegoModel(
        message_length=message_length,
        image_size=image_size,
        enable_distortions=True
    )
    model_with_dist.eval()

    with torch.no_grad():
        # Apply distortions manually
        from models.model import Distortions
        distortions = Distortions()
        distortions.train()  # Enable distortions

        distorted_stego = distortions(stego_image, apply_all=True)
        decoded_distorted = model_with_dist.decode(distorted_stego)

    accuracy_distorted = (decoded_distorted == binary_message).float().mean()
    print(
        f"   Bit accuracy after distortions: {accuracy_distorted.item()*100:.2f}%")

    print("\n6. Inference complete!")


def example_full_pipeline():
    """Example of the complete pipeline."""

    print("\n" + "=" * 70)
    print("EXAMPLE: Complete Pipeline (Encode → Distort → Decode)")
    print("=" * 70)

    # Configuration
    message_length = 512  # Smaller message for demo
    image_size = 128      # Smaller image for demo

    # Create model
    print("\n1. Creating model...")
    model = StegoModel(
        message_length=message_length,
        image_size=image_size,
        enable_distortions=True
    )

    # Prepare data
    print("\n2. Preparing data...")
    cover_image = torch.rand(2, 3, image_size, image_size)
    binary_message = torch.randint(0, 2, (2, message_length)).float()

    print(f"   Batch size: 2")
    print(f"   Message length: {message_length} bits")
    print(f"   Image size: {image_size}x{image_size}")

    # Run full pipeline
    print("\n3. Running full pipeline...")

    # With distortions (training mode)
    model.train()
    outputs_train = model.forward(cover_image, binary_message)

    print("   Training mode (with distortions):")
    print(
        f"      Stego image range: [{outputs_train['stego_image'].min():.3f}, {outputs_train['stego_image'].max():.3f}]")

    distortion_applied = (
        outputs_train['stego_image'] - outputs_train['distorted_stego']).abs().mean()
    print(f"      Distortion applied: {distortion_applied.item():.6f}")

    accuracy_train = (outputs_train['decoded_message']
                      == binary_message).float().mean()
    print(f"      Accuracy: {accuracy_train.item()*100:.2f}%")

    # Without distortions (eval mode)
    model.eval()
    with torch.no_grad():
        outputs_eval = model.forward(cover_image, binary_message)

    print("\n   Evaluation mode (no distortions):")
    distortion_applied = (
        outputs_eval['stego_image'] - outputs_eval['distorted_stego']).abs().mean()
    print(
        f"      Distortion applied: {distortion_applied.item():.6f} (should be 0)")

    accuracy_eval = (outputs_eval['decoded_message']
                     == binary_message).float().mean()
    print(f"      Accuracy: {accuracy_eval.item()*100:.2f}%")

    print("\n4. Pipeline complete!")


def example_save_load():
    """Example of saving and loading the model."""

    print("\n" + "=" * 70)
    print("EXAMPLE: Save and Load Model")
    print("=" * 70)

    # Create and save model
    print("\n1. Creating model...")
    model = StegoModel(message_length=1024, image_size=256)

    print("\n2. Saving model...")
    model.save_model("stego_model_checkpoint.pth")

    # Create new model and load
    print("\n3. Creating new model and loading weights...")
    new_model = StegoModel(message_length=1024, image_size=256)
    checkpoint = new_model.load_model("stego_model_checkpoint.pth")

    print(f"   Loaded checkpoint with:")
    print(f"      Message length: {checkpoint['message_length']}")
    print(f"      Image size: {checkpoint['image_size']}")

    # Verify models produce same output
    print("\n4. Verifying loaded model...")
    cover = torch.rand(1, 3, 256, 256)
    message = torch.randint(0, 2, (1, 1024)).float()

    with torch.no_grad():
        output1 = model.encode(cover, message)
        output2 = new_model.encode(cover, message)

    diff = (output1 - output2).abs().max()
    print(f"   Max difference between outputs: {diff.item():.10f}")
    print(f"   Models match: {diff.item() < 1e-6}")

    print("\n5. Save/load complete!")

    # Cleanup
    import os
    if os.path.exists("stego_model_checkpoint.pth"):
        os.remove("stego_model_checkpoint.pth")
        print("   Cleaned up checkpoint file.")


def calculate_psnr(img1, img2):
    """Calculate Peak Signal-to-Noise Ratio between two images."""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()


if __name__ == "__main__":
    print("\n")
    print("#" * 70)
    print("# StegoModel Usage Examples")
    print("#" * 70)

    # Run examples
    example_training()
    example_inference()
    example_full_pipeline()
    example_save_load()

    print("\n" + "#" * 70)
    print("# All examples completed successfully!")
    print("#" * 70)
    print("\n")
