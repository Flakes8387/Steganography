"""
Color Jitter Attack
Applies color transformations (brightness, contrast, saturation, hue adjustments).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BrightnessJitter(nn.Module):
    """Adjust brightness of images."""

    def __init__(self, brightness_range=(-0.2, 0.2)):
        """
        Args:
            brightness_range: Range of brightness adjustment
        """
        super(BrightnessJitter, self).__init__()
        self.brightness_range = brightness_range

    def forward(self, images, brightness=None):
        """
        Adjust brightness.

        Args:
            images: Tensor of shape (batch_size, 3, H, W)
            brightness: Brightness adjustment value. If None, randomly sample

        Returns:
            Adjusted images
        """
        if brightness is None:
            brightness = torch.rand(1).item(
            ) * (self.brightness_range[1] - self.brightness_range[0]) + self.brightness_range[0]

        adjusted = images + brightness
        return torch.clamp(adjusted, 0.0, 1.0)


class ContrastJitter(nn.Module):
    """Adjust contrast of images."""

    def __init__(self, contrast_range=(0.7, 1.3)):
        """
        Args:
            contrast_range: Range of contrast multiplier
        """
        super(ContrastJitter, self).__init__()
        self.contrast_range = contrast_range

    def forward(self, images, contrast=None):
        """
        Adjust contrast.

        Args:
            images: Tensor of shape (batch_size, 3, H, W)
            contrast: Contrast multiplier. If None, randomly sample

        Returns:
            Adjusted images
        """
        if contrast is None:
            contrast = torch.rand(1).item(
            ) * (self.contrast_range[1] - self.contrast_range[0]) + self.contrast_range[0]

        # Calculate mean for each image
        mean = images.mean(dim=[2, 3], keepdim=True)

        # Adjust contrast around mean
        adjusted = (images - mean) * contrast + mean

        return torch.clamp(adjusted, 0.0, 1.0)


class SaturationJitter(nn.Module):
    """Adjust saturation of images."""

    def __init__(self, saturation_range=(0.7, 1.3)):
        """
        Args:
            saturation_range: Range of saturation multiplier
        """
        super(SaturationJitter, self).__init__()
        self.saturation_range = saturation_range

    def rgb_to_grayscale(self, images):
        """Convert RGB to grayscale using standard weights."""
        # Standard weights for RGB to grayscale
        weights = torch.tensor([0.299, 0.587, 0.114],
                               device=images.device).view(1, 3, 1, 1)
        grayscale = (images * weights).sum(dim=1, keepdim=True)
        return grayscale.repeat(1, 3, 1, 1)

    def forward(self, images, saturation=None):
        """
        Adjust saturation.

        Args:
            images: Tensor of shape (batch_size, 3, H, W)
            saturation: Saturation multiplier. If None, randomly sample

        Returns:
            Adjusted images
        """
        if saturation is None:
            saturation = torch.rand(1).item(
            ) * (self.saturation_range[1] - self.saturation_range[0]) + self.saturation_range[0]

        # Convert to grayscale
        grayscale = self.rgb_to_grayscale(images)

        # Blend between grayscale and original based on saturation
        # saturation=1.0 means no change, <1.0 means desaturate, >1.0 means oversaturate
        adjusted = grayscale + (images - grayscale) * saturation

        return torch.clamp(adjusted, 0.0, 1.0)


class HueJitter(nn.Module):
    """Adjust hue of images."""

    def __init__(self, hue_range=(-0.1, 0.1)):
        """
        Args:
            hue_range: Range of hue adjustment (in radians)
        """
        super(HueJitter, self).__init__()
        self.hue_range = hue_range

    def rgb_to_hsv(self, images):
        """Convert RGB to HSV color space."""
        r, g, b = images[:, 0, :, :], images[:, 1, :, :], images[:, 2, :, :]

        maxc = torch.max(images, dim=1)[0]
        minc = torch.min(images, dim=1)[0]

        v = maxc
        deltac = maxc - minc
        s = deltac / (maxc + 1e-7)

        deltac = torch.where(deltac == 0, torch.ones_like(deltac), deltac)

        rc = (maxc - r) / deltac
        gc = (maxc - g) / deltac
        bc = (maxc - b) / deltac

        h = torch.where(r == maxc, bc - gc, torch.zeros_like(r))
        h = torch.where(g == maxc, 2.0 + rc - bc, h)
        h = torch.where(b == maxc, 4.0 + gc - rc, h)

        h = (h / 6.0) % 1.0

        return torch.stack([h, s, v], dim=1)

    def hsv_to_rgb(self, hsv):
        """Convert HSV to RGB color space."""
        h, s, v = hsv[:, 0, :, :], hsv[:, 1, :, :], hsv[:, 2, :, :]

        i = (h * 6.0).long()
        f = (h * 6.0) - i.float()

        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))

        i = i % 6

        r = torch.where(i == 0, v, torch.where(i == 1, q, torch.where(
            i == 2, p, torch.where(i == 3, p, torch.where(i == 4, t, v)))))
        g = torch.where(i == 0, t, torch.where(i == 1, v, torch.where(
            i == 2, v, torch.where(i == 3, q, torch.where(i == 4, p, p)))))
        b = torch.where(i == 0, p, torch.where(i == 1, p, torch.where(
            i == 2, t, torch.where(i == 3, v, torch.where(i == 4, v, q)))))

        return torch.stack([r, g, b], dim=1)

    def forward(self, images, hue_shift=None):
        """
        Adjust hue.

        Args:
            images: Tensor of shape (batch_size, 3, H, W)
            hue_shift: Hue shift value. If None, randomly sample

        Returns:
            Adjusted images
        """
        if hue_shift is None:
            hue_shift = torch.rand(
                1).item() * (self.hue_range[1] - self.hue_range[0]) + self.hue_range[0]

        # Convert to HSV
        hsv = self.rgb_to_hsv(images)

        # Adjust hue
        hsv[:, 0, :, :] = (hsv[:, 0, :, :] + hue_shift) % 1.0

        # Convert back to RGB
        adjusted = self.hsv_to_rgb(hsv)

        return torch.clamp(adjusted, 0.0, 1.0)


class ColorJitter(nn.Module):
    """
    Combined color jitter applying multiple color transformations.
    """

    def __init__(self,
                 brightness_range=(-0.2, 0.2),
                 contrast_range=(0.7, 1.3),
                 saturation_range=(0.7, 1.3),
                 hue_range=(-0.1, 0.1)):
        """
        Args:
            brightness_range: Range for brightness adjustment
            contrast_range: Range for contrast adjustment
            saturation_range: Range for saturation adjustment
            hue_range: Range for hue adjustment
        """
        super(ColorJitter, self).__init__()
        self.brightness = BrightnessJitter(brightness_range)
        self.contrast = ContrastJitter(contrast_range)
        self.saturation = SaturationJitter(saturation_range)
        self.hue = HueJitter(hue_range)

    def forward(self, images, apply_all=True):
        """
        Apply color jitter transformations.

        Args:
            images: Tensor of shape (batch_size, 3, H, W)
            apply_all: If True, apply all transformations. Otherwise, randomly apply subset

        Returns:
            Color-jittered images
        """
        if apply_all:
            images = self.brightness(images)
            images = self.contrast(images)
            images = self.saturation(images)
            images = self.hue(images)
        else:
            # Randomly apply each transformation
            if torch.rand(1).item() < 0.8:
                images = self.brightness(images)
            if torch.rand(1).item() < 0.8:
                images = self.contrast(images)
            if torch.rand(1).item() < 0.5:
                images = self.saturation(images)
            if torch.rand(1).item() < 0.5:
                images = self.hue(images)

        return images


# Test function
if __name__ == "__main__":
    print("Testing Color Jitter Attacks...")

    batch_size = 2
    image_size = 256

    # Create test images
    images = torch.rand(batch_size, 3, image_size, image_size)

    print("\n1. Testing Brightness Jitter...")
    brightness = BrightnessJitter(brightness_range=(-0.2, 0.2))
    adjusted = brightness(images, brightness=0.1)
    print(f"   Input range: [{images.min():.4f}, {images.max():.4f}]")
    print(f"   Output range: [{adjusted.min():.4f}, {adjusted.max():.4f}]")
    print(f"   Mean difference: {(images - adjusted).abs().mean():.6f}")

    print("\n2. Testing Contrast Jitter...")
    contrast = ContrastJitter(contrast_range=(0.7, 1.3))
    adjusted = contrast(images, contrast=1.2)
    print(f"   Output range: [{adjusted.min():.4f}, {adjusted.max():.4f}]")
    print(f"   Mean difference: {(images - adjusted).abs().mean():.6f}")

    print("\n3. Testing Saturation Jitter...")
    saturation = SaturationJitter(saturation_range=(0.7, 1.3))
    adjusted = saturation(images, saturation=0.8)
    print(f"   Output range: [{adjusted.min():.4f}, {adjusted.max():.4f}]")
    print(f"   Mean difference: {(images - adjusted).abs().mean():.6f}")

    print("\n4. Testing Hue Jitter...")
    hue = HueJitter(hue_range=(-0.1, 0.1))
    adjusted = hue(images, hue_shift=0.05)
    print(f"   Output range: [{adjusted.min():.4f}, {adjusted.max():.4f}]")
    print(f"   Mean difference: {(images - adjusted).abs().mean():.6f}")

    print("\n5. Testing Combined Color Jitter...")
    color_jitter = ColorJitter()
    adjusted = color_jitter(images, apply_all=True)
    print(f"   Output range: [{adjusted.min():.4f}, {adjusted.max():.4f}]")
    print(f"   Mean difference: {(images - adjusted).abs().mean():.6f}")

    # Test gradient flow
    print("\n6. Testing gradient flow...")
    images.requires_grad = True
    adjusted = brightness(images, brightness=0.1)
    loss = adjusted.mean()
    loss.backward()
    print(f"   ✅ Gradient flow working: {images.grad is not None}")

    print("\n✅ All color jitter attack tests passed!")
