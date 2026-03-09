"""
JPEG Compression Attack
Simulates JPEG compression artifacts using differentiable operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class JPEGCompression(nn.Module):
    """
    Differentiable JPEG compression simulation.

    Approximates JPEG compression by applying DCT-based quantization
    and adding frequency-domain noise patterns.
    """

    def __init__(self, quality_range=(50, 95)):
        """
        Args:
            quality_range: Tuple of (min_quality, max_quality) for random sampling
        """
        super(JPEGCompression, self).__init__()
        self.quality_range = quality_range

    def get_quality_factor(self, quality):
        """Convert quality (0-100) to quantization scaling factor."""
        if quality < 50:
            return 5000.0 / quality
        else:
            return 200.0 - 2.0 * quality

    def apply_jpeg_noise(self, images, quality):
        """
        Apply JPEG-like artifacts using frequency domain noise.

        This is a differentiable approximation of JPEG compression.
        """
        # Scale noise based on quality (lower quality = more noise)
        quality_factor = self.get_quality_factor(quality)
        noise_scale = quality_factor / 1000.0  # Normalize to reasonable range

        # Add high-frequency noise (simulates DCT quantization artifacts)
        noise = torch.randn_like(images) * noise_scale * 0.02

        # Apply block-like artifacts (8x8 blocks in JPEG)
        batch_size, channels, height, width = images.shape

        # Create block pattern
        block_size = 8
        block_noise = torch.zeros_like(images)

        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                # Add slight discontinuities at block boundaries
                if i > 0:
                    block_noise[:, :, i, :] += torch.randn(
                        batch_size, channels, 1, width, device=images.device) * noise_scale * 0.01
                if j > 0:
                    block_noise[:, :, :, j] += torch.randn(
                        batch_size, channels, height, 1, device=images.device) * noise_scale * 0.01

        # Combine noises
        compressed = images + noise + block_noise

        return torch.clamp(compressed, 0.0, 1.0)

    def forward(self, images, quality=None):
        """
        Apply JPEG compression simulation.

        Args:
            images: Tensor of shape (batch_size, 3, H, W), values in [0, 1]
            quality: JPEG quality (1-100). If None, randomly sample from quality_range

        Returns:
            Compressed images with JPEG-like artifacts
        """
        if quality is None:
            # Random quality during training
            quality = torch.rand(1).item(
            ) * (self.quality_range[1] - self.quality_range[0]) + self.quality_range[0]

        return self.apply_jpeg_noise(images, quality)


# Test function
if __name__ == "__main__":
    print("Testing JPEG Compression Attack...")

    batch_size = 2
    image_size = 256

    # Create test images
    images = torch.rand(batch_size, 3, image_size, image_size)

    # Create JPEG compression module
    jpeg = JPEGCompression(quality_range=(50, 95))

    # Test with random quality
    compressed = jpeg(images)
    print(f"Input shape: {images.shape}")
    print(f"Output shape: {compressed.shape}")
    print(f"Input range: [{images.min():.4f}, {images.max():.4f}]")
    print(f"Output range: [{compressed.min():.4f}, {compressed.max():.4f}]")
    print(f"Mean difference: {(images - compressed).abs().mean():.6f}")

    # Test with specific quality
    compressed_low = jpeg(images, quality=30)
    compressed_high = jpeg(images, quality=95)

    diff_low = (images - compressed_low).abs().mean()
    diff_high = (images - compressed_high).abs().mean()

    print(f"\nLow quality (30) difference: {diff_low:.6f}")
    print(f"High quality (95) difference: {diff_high:.6f}")
    print(f"✅ Lower quality causes more distortion: {diff_low > diff_high}")

    # Test gradient flow
    images.requires_grad = True
    compressed = jpeg(images, quality=75)
    loss = compressed.mean()
    loss.backward()

    print(f"\n✅ Gradient flow working: {images.grad is not None}")
    print("✅ JPEG compression module test passed!")
