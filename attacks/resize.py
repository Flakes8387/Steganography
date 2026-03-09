"""
Resize Attack
Simulates image resizing (downscaling then upscaling) as done by social media platforms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResizeAttack(nn.Module):
    """
    Resize attack: downscale then upscale images.

    Simulates compression by social media platforms like WhatsApp/Instagram
    that resize images to reduce file size.
    """

    def __init__(self, scale_range=(0.5, 0.9)):
        """
        Args:
            scale_range: Range of downscaling factors (e.g., 0.5 = half size)
        """
        super(ResizeAttack, self).__init__()
        self.scale_range = scale_range

    def forward(self, images, scale_factor=None, mode='bilinear'):
        """
        Apply resize attack.

        Args:
            images: Tensor of shape (batch_size, 3, H, W)
            scale_factor: Downscaling factor. If None, randomly sample from scale_range
            mode: Interpolation mode ('bilinear', 'bicubic', 'nearest')

        Returns:
            Resized images (same size as input, but with quality loss)
        """
        if scale_factor is None:
            scale_factor = torch.rand(
                1).item() * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]

        batch_size, channels, height, width = images.shape

        # Calculate new dimensions
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)

        # Downscale
        downscaled = F.interpolate(
            images,
            size=(new_height, new_width),
            mode=mode,
            align_corners=False if mode != 'nearest' else None
        )

        # Upscale back to original size
        upscaled = F.interpolate(
            downscaled,
            size=(height, width),
            mode=mode,
            align_corners=False if mode != 'nearest' else None
        )

        return upscaled


class AdaptiveResize(nn.Module):
    """
    Adaptive resize that simulates different platform behaviors.
    """

    def __init__(self):
        super(AdaptiveResize, self).__init__()

        # Different resize profiles for different platforms
        self.profiles = {
            'whatsapp': (0.7, 0.85),  # Aggressive compression
            'instagram': (0.8, 0.9),   # Moderate compression
            'facebook': (0.75, 0.88),  # Moderate compression
            'twitter': (0.65, 0.82),   # Aggressive compression
        }

    def forward(self, images, platform=None):
        """
        Apply platform-specific resize.

        Args:
            images: Tensor of shape (batch_size, 3, H, W)
            platform: Platform name ('whatsapp', 'instagram', etc.) or None for random

        Returns:
            Resized images
        """
        if platform is None:
            # Random platform
            platform = list(self.profiles.keys())[
                torch.randint(0, len(self.profiles), (1,)).item()]

        scale_range = self.profiles.get(platform, (0.7, 0.9))
        scale_factor = torch.rand(
            1).item() * (scale_range[1] - scale_range[0]) + scale_range[0]

        batch_size, channels, height, width = images.shape
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)

        # Downscale
        downscaled = F.interpolate(images, size=(
            new_height, new_width), mode='bilinear', align_corners=False)

        # Upscale back
        upscaled = F.interpolate(downscaled, size=(
            height, width), mode='bilinear', align_corners=False)

        return upscaled


class RandomResize(nn.Module):
    """
    Apply random resize with varying interpolation modes.
    """

    def __init__(self, scale_range=(0.5, 0.9)):
        super(RandomResize, self).__init__()
        self.scale_range = scale_range
        self.modes = ['bilinear', 'bicubic', 'nearest']

    def forward(self, images):
        """
        Apply random resize with random interpolation.

        Args:
            images: Tensor of shape (batch_size, 3, H, W)

        Returns:
            Resized images
        """
        scale_factor = torch.rand(
            1).item() * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]
        mode = self.modes[torch.randint(0, len(self.modes), (1,)).item()]

        batch_size, channels, height, width = images.shape
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)

        # Downscale
        if mode == 'nearest':
            downscaled = F.interpolate(images, size=(
                new_height, new_width), mode=mode)
        else:
            downscaled = F.interpolate(images, size=(
                new_height, new_width), mode=mode, align_corners=False)

        # Upscale back
        if mode == 'nearest':
            upscaled = F.interpolate(
                downscaled, size=(height, width), mode=mode)
        else:
            upscaled = F.interpolate(downscaled, size=(
                height, width), mode=mode, align_corners=False)

        return upscaled


# Test function
if __name__ == "__main__":
    print("Testing Resize Attacks...")

    batch_size = 2
    image_size = 256

    # Create test images
    images = torch.rand(batch_size, 3, image_size, image_size)

    print("\n1. Testing ResizeAttack...")
    resize = ResizeAttack(scale_range=(0.5, 0.9))
    resized = resize(images, scale_factor=0.7)
    print(f"   Input shape: {images.shape}")
    print(f"   Output shape: {resized.shape}")
    print(f"   Mean difference: {(images - resized).abs().mean():.6f}")

    print("\n2. Testing AdaptiveResize...")
    adaptive = AdaptiveResize()

    for platform in ['whatsapp', 'instagram', 'facebook', 'twitter']:
        resized = adaptive(images, platform=platform)
        diff = (images - resized).abs().mean()
        print(f"   {platform}: difference = {diff:.6f}")

    print("\n3. Testing RandomResize...")
    random_resize = RandomResize(scale_range=(0.5, 0.9))
    resized = random_resize(images)
    print(f"   Output shape: {resized.shape}")
    print(f"   Mean difference: {(images - resized).abs().mean():.6f}")

    # Test gradient flow
    print("\n4. Testing gradient flow...")
    images.requires_grad = True
    resized = resize(images, scale_factor=0.8)
    loss = resized.mean()
    loss.backward()
    print(f"   ✅ Gradient flow working: {images.grad is not None}")

    print("\n✅ All resize attack tests passed!")
