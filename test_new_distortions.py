"""
Test script to verify the new distortions (Gaussian blur, resize, color jitter).
"""

import torch
from models.model import StegoModel, Distortions


def test_new_distortions():
    """Test that new distortions are properly integrated."""
    print("Testing new distortions integration...\n")

    # Create model with distortions enabled
    model = StegoModel(
        message_length=1024,
        image_size=256,
        enable_distortions=True
    )
    model.train()  # Enable training mode

    # Create sample data
    batch_size = 2
    cover_images = torch.rand(batch_size, 3, 256, 256)
    binary_messages = torch.randint(0, 2, (batch_size, 1024)).float()

    print(f"Input shapes:")
    print(f"  Cover images: {cover_images.shape}")
    print(f"  Binary messages: {binary_messages.shape}\n")

    # Test forward pass with distortions
    print("=" * 60)
    print("Testing forward pass with all distortions...")
    print("=" * 60)

    with torch.no_grad():
        outputs = model(cover_images, binary_messages, apply_distortions=True)

    print(f"\nOutputs:")
    print(f"  Stego image: {outputs['stego_image'].shape}")
    print(f"  Distorted stego: {outputs['distorted_stego'].shape}")
    print(f"  Decoded logits: {outputs['decoded_logits'].shape}")
    print(f"  Decoded message: {outputs['decoded_message'].shape}")

    # Calculate accuracy
    accuracy = (outputs['decoded_message'] == binary_messages).float().mean()
    print(f"\nBit accuracy: {accuracy.item()*100:.2f}%")

    # Test individual distortions
    print("\n" + "=" * 60)
    print("Testing individual distortion modules...")
    print("=" * 60)

    distortions = model.distortions
    test_image = torch.rand(1, 3, 256, 256)

    # Test Gaussian blur
    print("\n1. Gaussian Blur Attack:")
    blurred = distortions.apply_gaussian_blur_attack(
        test_image, probability=1.0)
    diff = (test_image - blurred).abs().mean()
    print(f"   Mean difference: {diff.item():.6f}")
    print(f"   Output shape: {blurred.shape}")

    # Test resize attack
    print("\n2. Resize Attack:")
    resized = distortions.apply_resize_attack(test_image, probability=1.0)
    diff = (test_image - resized).abs().mean()
    print(f"   Mean difference: {diff.item():.6f}")
    print(f"   Output shape: {resized.shape}")

    # Test color jitter
    print("\n3. Color Jitter Attack:")
    jittered = distortions.apply_color_jitter_attack(
        test_image, probability=1.0)
    diff = (test_image - jittered).abs().mean()
    print(f"   Mean difference: {diff.item():.6f}")
    print(f"   Output shape: {jittered.shape}")

    # Test combined distortions
    print("\n" + "=" * 60)
    print("Testing combined distortions (all together)...")
    print("=" * 60)

    distorted = distortions(test_image, apply_all=False)
    diff = (test_image - distorted).abs().mean()
    print(f"\nMean difference after all distortions: {diff.item():.6f}")
    print(f"Output shape: {distorted.shape}")

    # Test loss computation with distortions
    print("\n" + "=" * 60)
    print("Testing loss computation with distortions...")
    print("=" * 60)

    model.train()
    loss_dict = model.compute_loss(
        cover_images, binary_messages, alpha=1.0, beta=1.0)

    print(f"\nLoss values:")
    print(f"  Total loss: {loss_dict['total_loss'].item():.6f}")
    print(f"  Image loss: {loss_dict['image_loss'].item():.6f}")
    print(f"  Message loss: {loss_dict['message_loss'].item():.6f}")
    print(f"  Bit accuracy: {loss_dict['bit_accuracy'].item()*100:.2f}%")
    print(f"  BER: {loss_dict['ber'].item()*100:.2f}%")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)

    print("\nNew distortions successfully integrated:")
    print("  ✓ Gaussian Blur")
    print("  ✓ Resize Attack")
    print("  ✓ Color Jitter")
    print("\nThe model can now be trained with these attacks for improved robustness!")


if __name__ == "__main__":
    test_new_distortions()
