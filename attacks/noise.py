"""
Noise Attack
Adds various types of noise to images (Gaussian, salt-and-pepper, speckle).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianNoise(nn.Module):
    """Add Gaussian noise to images."""

    def __init__(self, std_range=(0.01, 0.05)):
        """
        Args:
            std_range: Range of standard deviation for noise
        """
        super(GaussianNoise, self).__init__()
        self.std_range = std_range

    def forward(self, images, std=None):
        """
        Add Gaussian noise.

        Args:
            images: Tensor of shape (batch_size, 3, H, W)
            std: Standard deviation. If None, randomly sample from std_range

        Returns:
            Noisy images
        """
        if std is None:
            std = torch.rand(
                1).item() * (self.std_range[1] - self.std_range[0]) + self.std_range[0]

        noise = torch.randn_like(images) * std
        noisy_images = images + noise

        return torch.clamp(noisy_images, 0.0, 1.0)


class SaltPepperNoise(nn.Module):
    """Add salt-and-pepper noise (random white and black pixels)."""

    def __init__(self, prob_range=(0.01, 0.05)):
        """
        Args:
            prob_range: Range of probability for salt-and-pepper noise
        """
        super(SaltPepperNoise, self).__init__()
        self.prob_range = prob_range

    def forward(self, images, prob=None):
        """
        Add salt-and-pepper noise.

        Args:
            images: Tensor of shape (batch_size, 3, H, W)
            prob: Probability of noise. If None, randomly sample from prob_range

        Returns:
            Noisy images
        """
        if prob is None:
            prob = torch.rand(
                1).item() * (self.prob_range[1] - self.prob_range[0]) + self.prob_range[0]

        # Create noise mask
        noise_mask = torch.rand_like(images)

        # Salt (white pixels)
        salt_mask = noise_mask < prob / 2

        # Pepper (black pixels)
        pepper_mask = (noise_mask >= prob / 2) & (noise_mask < prob)

        noisy_images = images.clone()
        noisy_images[salt_mask] = 1.0
        noisy_images[pepper_mask] = 0.0

        return noisy_images


class SpeckleNoise(nn.Module):
    """Add speckle noise (multiplicative noise)."""

    def __init__(self, std_range=(0.01, 0.05)):
        """
        Args:
            std_range: Range of standard deviation for speckle noise
        """
        super(SpeckleNoise, self).__init__()
        self.std_range = std_range

    def forward(self, images, std=None):
        """
        Add speckle noise.

        Args:
            images: Tensor of shape (batch_size, 3, H, W)
            std: Standard deviation. If None, randomly sample from std_range

        Returns:
            Noisy images
        """
        if std is None:
            std = torch.rand(
                1).item() * (self.std_range[1] - self.std_range[0]) + self.std_range[0]

        noise = torch.randn_like(images) * std
        noisy_images = images * (1 + noise)

        return torch.clamp(noisy_images, 0.0, 1.0)


class CombinedNoise(nn.Module):
    """Randomly apply one of several noise types."""

    def __init__(self):
        super(CombinedNoise, self).__init__()
        self.gaussian = GaussianNoise()
        self.salt_pepper = SaltPepperNoise()
        self.speckle = SpeckleNoise()

    def forward(self, images):
        """
        Randomly apply one type of noise.

        Args:
            images: Tensor of shape (batch_size, 3, H, W)

        Returns:
            Noisy images
        """
        noise_type = torch.rand(1).item()

        if noise_type < 0.4:
            return self.gaussian(images)
        elif noise_type < 0.7:
            return self.speckle(images)
        else:
            return self.salt_pepper(images)


# Test function
if __name__ == "__main__":
    print("Testing Noise Attacks...")

    batch_size = 2
    image_size = 256

    # Create test images
    images = torch.rand(batch_size, 3, image_size, image_size)

    print("\n1. Testing Gaussian Noise...")
    gaussian = GaussianNoise(std_range=(0.01, 0.05))
    noisy = gaussian(images)
    print(f"   Input range: [{images.min():.4f}, {images.max():.4f}]")
    print(f"   Output range: [{noisy.min():.4f}, {noisy.max():.4f}]")
    print(f"   Mean difference: {(images - noisy).abs().mean():.6f}")

    print("\n2. Testing Salt-Pepper Noise...")
    salt_pepper = SaltPepperNoise(prob_range=(0.01, 0.05))
    noisy = salt_pepper(images)
    print(f"   Output range: [{noisy.min():.4f}, {noisy.max():.4f}]")
    print(f"   Pixels changed: {(images != noisy).float().mean():.4f}")

    print("\n3. Testing Speckle Noise...")
    speckle = SpeckleNoise(std_range=(0.01, 0.05))
    noisy = speckle(images)
    print(f"   Output range: [{noisy.min():.4f}, {noisy.max():.4f}]")
    print(f"   Mean difference: {(images - noisy).abs().mean():.6f}")

    print("\n4. Testing Combined Noise...")
    combined = CombinedNoise()
    noisy = combined(images)
    print(f"   Output range: [{noisy.min():.4f}, {noisy.max():.4f}]")

    # Test gradient flow
    print("\n5. Testing gradient flow...")
    images.requires_grad = True
    noisy = gaussian(images)
    loss = noisy.mean()
    loss.backward()
    print(f"   ✅ Gradient flow working: {images.grad is not None}")

    print("\n✅ All noise attack tests passed!")
