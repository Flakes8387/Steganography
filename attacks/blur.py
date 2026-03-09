"""
Blur Attack
Applies various blur operations (Gaussian blur, motion blur, defocus blur).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GaussianBlur(nn.Module):
    """Apply Gaussian blur using 2D convolution."""

    def __init__(self, kernel_size_range=(3, 7), sigma_range=(0.5, 2.0)):
        """
        Args:
            kernel_size_range: Range of kernel sizes (must be odd)
            sigma_range: Range of sigma values for Gaussian kernel
        """
        super(GaussianBlur, self).__init__()
        self.kernel_size_range = kernel_size_range
        self.sigma_range = sigma_range

    def get_gaussian_kernel(self, kernel_size, sigma):
        """Generate 2D Gaussian kernel."""
        # Create coordinate grid
        coords = torch.arange(kernel_size, dtype=torch.float32)
        coords = coords - kernel_size // 2

        # Create 2D grid
        x, y = torch.meshgrid(coords, coords, indexing='ij')

        # Gaussian formula
        kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()

        return kernel

    def forward(self, images, kernel_size=None, sigma=None):
        """
        Apply Gaussian blur.

        Args:
            images: Tensor of shape (batch_size, 3, H, W)
            kernel_size: Size of Gaussian kernel (odd number)
            sigma: Standard deviation of Gaussian

        Returns:
            Blurred images
        """
        if kernel_size is None:
            kernel_size = torch.randint(
                self.kernel_size_range[0],
                self.kernel_size_range[1] + 1,
                (1,)
            ).item()
            # Ensure odd
            kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

        if sigma is None:
            sigma = torch.rand(
                1).item() * (self.sigma_range[1] - self.sigma_range[0]) + self.sigma_range[0]

        # Generate Gaussian kernel
        kernel = self.get_gaussian_kernel(kernel_size, sigma)
        kernel = kernel.to(images.device)

        # Expand kernel for convolution (out_channels, in_channels/groups, kH, kW)
        kernel = kernel.view(1, 1, kernel_size, kernel_size)
        kernel = kernel.repeat(3, 1, 1, 1)  # 3 channels

        # Apply convolution (same padding)
        padding = kernel_size // 2
        blurred = F.conv2d(images, kernel, padding=padding, groups=3)

        return blurred


class MotionBlur(nn.Module):
    """Apply motion blur using directional kernel."""

    def __init__(self, kernel_size_range=(5, 15)):
        """
        Args:
            kernel_size_range: Range of kernel sizes
        """
        super(MotionBlur, self).__init__()
        self.kernel_size_range = kernel_size_range

    def get_motion_kernel(self, kernel_size, angle):
        """Generate motion blur kernel at given angle."""
        kernel = torch.zeros((kernel_size, kernel_size))

        # Center of kernel
        center = kernel_size // 2

        # Convert angle to radians
        angle_rad = angle * math.pi / 180.0

        # Create line along angle
        for i in range(kernel_size):
            offset = i - center
            x = center + int(offset * math.cos(angle_rad))
            y = center + int(offset * math.sin(angle_rad))

            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                kernel[x, y] = 1.0

        # Normalize
        kernel = kernel / kernel.sum() if kernel.sum() > 0 else kernel

        return kernel

    def forward(self, images, kernel_size=None, angle=None):
        """
        Apply motion blur.

        Args:
            images: Tensor of shape (batch_size, 3, H, W)
            kernel_size: Size of motion kernel
            angle: Angle of motion (0-360 degrees)

        Returns:
            Blurred images
        """
        if kernel_size is None:
            kernel_size = torch.randint(
                self.kernel_size_range[0],
                self.kernel_size_range[1] + 1,
                (1,)
            ).item()
            # Ensure odd
            kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

        if angle is None:
            angle = torch.rand(1).item() * 360.0

        # Generate motion kernel
        kernel = self.get_motion_kernel(kernel_size, angle)
        kernel = kernel.to(images.device)

        # Expand kernel for convolution
        kernel = kernel.view(1, 1, kernel_size, kernel_size)
        kernel = kernel.repeat(3, 1, 1, 1)

        # Apply convolution
        padding = kernel_size // 2
        blurred = F.conv2d(images, kernel, padding=padding, groups=3)

        return blurred


class AverageBlur(nn.Module):
    """Apply average/box blur."""

    def __init__(self, kernel_size_range=(3, 7)):
        """
        Args:
            kernel_size_range: Range of kernel sizes (must be odd)
        """
        super(AverageBlur, self).__init__()
        self.kernel_size_range = kernel_size_range

    def forward(self, images, kernel_size=None):
        """
        Apply average blur.

        Args:
            images: Tensor of shape (batch_size, 3, H, W)
            kernel_size: Size of averaging kernel

        Returns:
            Blurred images
        """
        if kernel_size is None:
            kernel_size = torch.randint(
                self.kernel_size_range[0],
                self.kernel_size_range[1] + 1,
                (1,)
            ).item()
            # Ensure odd
            kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

        # Create uniform kernel
        kernel = torch.ones((kernel_size, kernel_size), device=images.device)
        kernel = kernel / kernel.sum()

        # Expand kernel for convolution
        kernel = kernel.view(1, 1, kernel_size, kernel_size)
        kernel = kernel.repeat(3, 1, 1, 1)

        # Apply convolution
        padding = kernel_size // 2
        blurred = F.conv2d(images, kernel, padding=padding, groups=3)

        return blurred


class CombinedBlur(nn.Module):
    """Randomly apply one of several blur types."""

    def __init__(self):
        super(CombinedBlur, self).__init__()
        self.gaussian = GaussianBlur()
        self.motion = MotionBlur()
        self.average = AverageBlur()

    def forward(self, images):
        """
        Randomly apply one type of blur.

        Args:
            images: Tensor of shape (batch_size, 3, H, W)

        Returns:
            Blurred images
        """
        blur_type = torch.rand(1).item()

        if blur_type < 0.5:
            return self.gaussian(images)
        elif blur_type < 0.8:
            return self.average(images)
        else:
            return self.motion(images)


# Test function
if __name__ == "__main__":
    print("Testing Blur Attacks...")

    batch_size = 2
    image_size = 256

    # Create test images
    images = torch.rand(batch_size, 3, image_size, image_size)

    print("\n1. Testing Gaussian Blur...")
    gaussian = GaussianBlur(kernel_size_range=(3, 7), sigma_range=(0.5, 2.0))
    blurred = gaussian(images, kernel_size=5, sigma=1.0)
    print(f"   Input shape: {images.shape}")
    print(f"   Output shape: {blurred.shape}")
    print(f"   Mean difference: {(images - blurred).abs().mean():.6f}")

    print("\n2. Testing Motion Blur...")
    motion = MotionBlur(kernel_size_range=(5, 15))
    blurred = motion(images, kernel_size=9, angle=45)
    print(f"   Output shape: {blurred.shape}")
    print(f"   Mean difference: {(images - blurred).abs().mean():.6f}")

    print("\n3. Testing Average Blur...")
    average = AverageBlur(kernel_size_range=(3, 7))
    blurred = average(images, kernel_size=5)
    print(f"   Output shape: {blurred.shape}")
    print(f"   Mean difference: {(images - blurred).abs().mean():.6f}")

    print("\n4. Testing Combined Blur...")
    combined = CombinedBlur()
    blurred = combined(images)
    print(f"   Output shape: {blurred.shape}")

    # Test gradient flow
    print("\n5. Testing gradient flow...")
    images.requires_grad = True
    blurred = gaussian(images, kernel_size=5, sigma=1.0)
    loss = blurred.mean()
    loss.backward()
    print(f"   ✅ Gradient flow working: {images.grad is not None}")

    print("\n✅ All blur attack tests passed!")
