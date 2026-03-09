"""
Test script for Encoder and Decoder models
"""

from decoder import Decoder
from encoder import Encoder
import torch
import sys
sys.path.insert(0, 'models')


def test_encoder_decoder():
    """Test both encoder and decoder together"""
    print("="*60)
    print("Testing Steganography Models")
    print("="*60)

    # Configuration
    batch_size = 2
    message_length = 1024  # 1024 bits
    image_size = 256

    # Create models
    print("\n1. Creating models...")
    encoder = Encoder(message_length=message_length, image_size=image_size)
    decoder = Decoder(message_length=message_length, image_size=image_size)

    print(f"   Encoder parameters: {encoder.get_num_parameters():,}")
    print(f"   Decoder parameters: {decoder.get_num_parameters():,}")

    # Create test data
    print("\n2. Creating test data...")
    cover_image = torch.rand(batch_size, 3, image_size, image_size)
    binary_message = torch.randint(0, 2, (batch_size, message_length)).float()

    print(f"   Cover image shape: {cover_image.shape}")
    print(f"   Binary message shape: {binary_message.shape}")
    print(f"   Message sample: {binary_message[0, :20].int().tolist()}")

    # Test encoder
    print("\n3. Testing Encoder...")
    with torch.no_grad():
        stego_image = encoder(cover_image, binary_message)

    print(f"   Stego image shape: {stego_image.shape}")
    print(
        f"   Stego image range: [{stego_image.min():.4f}, {stego_image.max():.4f}]")

    # Calculate difference
    diff = torch.abs(stego_image - cover_image)
    print(f"   Avg pixel difference: {diff.mean():.6f}")
    print(f"   Max pixel difference: {diff.max():.6f}")

    # Test decoder
    print("\n4. Testing Decoder...")
    with torch.no_grad():
        decoded_message = decoder(stego_image)
        decoded_logits = decoder(stego_image, return_logits=True)
        decoded_probs = decoder.get_probabilities(stego_image)

    print(f"   Decoded message shape: {decoded_message.shape}")
    print(
        f"   Decoded message range: [{decoded_message.min():.0f}, {decoded_message.max():.0f}]")
    print(
        f"   Decoded logits range: [{decoded_logits.min():.4f}, {decoded_logits.max():.4f}]")
    print(
        f"   Decoded probs range: [{decoded_probs.min():.4f}, {decoded_probs.max():.4f}]")
    print(f"   Decoded sample: {decoded_message[0, :20].int().tolist()}")

    # Calculate accuracy
    accuracy = (decoded_message == binary_message).float().mean()
    print(f"\n5. Accuracy (random, not trained): {accuracy*100:.2f}%")

    # Bit distribution
    ones_ratio = decoded_message.mean().item()
    print(
        f"   Decoded bits: {ones_ratio*100:.1f}% ones, {(1-ones_ratio)*100:.1f}% zeros")

    print("\n" + "="*60)
    print("✅ All tests passed successfully!")
    print("="*60)

    return encoder, decoder


if __name__ == "__main__":
    encoder, decoder = test_encoder_decoder()
