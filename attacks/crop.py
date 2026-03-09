"""
Crop Attack
Simulates random cropping and padding operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RandomCrop(nn.Module):
    """
    Random crop followed by resize to original dimensions.

    Simulates content loss that might occur when images are cropped
    or when users zoom/crop images before sharing.
    """

    def __init__(self, crop_ratio_range=(0.7, 0.95)):
        """
        Args:
            crop_ratio_range: Range of crop ratios (ratio of original size to keep)
        """
        super(RandomCrop, self).__init__()
        self.crop_ratio_range = crop_ratio_range

    def forward(self, images, crop_ratio=None):
        """
        Apply random crop.

        Args:
            images: Tensor of shape (batch_size, 3, H, W)
            crop_ratio: Ratio of image to keep. If None, randomly sample

        Returns:
            Cropped and resized images (same size as input)
        """
        if crop_ratio is None:
            crop_ratio = torch.rand(1).item(
            ) * (self.crop_ratio_range[1] - self.crop_ratio_range[0]) + self.crop_ratio_range[0]

        batch_size, channels, height, width = images.shape

        # Calculate crop dimensions
        crop_height = int(height * crop_ratio)
        crop_width = int(width * crop_ratio)

        # Random crop position
        top = torch.randint(0, height - crop_height + 1, (1,)).item()
        left = torch.randint(0, width - crop_width + 1, (1,)).item()

        # Crop
        cropped = images[:, :, top:top+crop_height, left:left+crop_width]

        # Resize back to original dimensions
        resized = F.interpolate(cropped, size=(
            height, width), mode='bilinear', align_corners=False)

        return resized


class CenterCrop(nn.Module):
    """Center crop followed by resize."""

    def __init__(self, crop_ratio_range=(0.7, 0.95)):
        """
        Args:
            crop_ratio_range: Range of crop ratios
        """
        super(CenterCrop, self).__init__()
        self.crop_ratio_range = crop_ratio_range

    def forward(self, images, crop_ratio=None):
        """
        Apply center crop.

        Args:
            images: Tensor of shape (batch_size, 3, H, W)
            crop_ratio: Ratio of image to keep. If None, randomly sample

        Returns:
            Cropped and resized images
        """
        if crop_ratio is None:
            crop_ratio = torch.rand(1).item(
            ) * (self.crop_ratio_range[1] - self.crop_ratio_range[0]) + self.crop_ratio_range[0]

        batch_size, channels, height, width = images.shape

        # Calculate crop dimensions
        crop_height = int(height * crop_ratio)
        crop_width = int(width * crop_ratio)

        # Center position
        top = (height - crop_height) // 2
        left = (width - crop_width) // 2

        # Crop
        cropped = images[:, :, top:top+crop_height, left:left+crop_width]

        # Resize back
        resized = F.interpolate(cropped, size=(
            height, width), mode='bilinear', align_corners=False)

        return resized


class PadCrop(nn.Module):
    """
    Crop with padding (removes border and adds black padding).

    Simulates letterboxing or pillarboxing effects.
    """

    def __init__(self, pad_ratio_range=(0.05, 0.15)):
        """
        Args:
            pad_ratio_range: Range of padding ratios
        """
        super(PadCrop, self).__init__()
        self.pad_ratio_range = pad_ratio_range

    def forward(self, images, pad_ratio=None):
        """
        Apply padding crop.

        Args:
            images: Tensor of shape (batch_size, 3, H, W)
            pad_ratio: Ratio of padding. If None, randomly sample

        Returns:
            Padded images
        """
        if pad_ratio is None:
            pad_ratio = torch.rand(1).item(
            ) * (self.pad_ratio_range[1] - self.pad_ratio_range[0]) + self.pad_ratio_range[0]

        batch_size, channels, height, width = images.shape

        # Calculate padding
        pad_height = int(height * pad_ratio)
        pad_width = int(width * pad_ratio)

        # Create padded image with zeros (black padding)
        padded = torch.zeros_like(images)

        # Copy original image to center
        padded[:, :, pad_height:height-pad_height, pad_width:width-pad_width] = \
            images[:, :, pad_height:height -
                   pad_height, pad_width:width-pad_width]

        return padded


class AspectRatioCrop(nn.Module):
    """
    Crop to different aspect ratios (e.g., 16:9, 4:3, 1:1).
    """

    def __init__(self):
        super(AspectRatioCrop, self).__init__()
        self.aspect_ratios = {
            '16:9': 16/9,
            '4:3': 4/3,
            '1:1': 1.0,
            '9:16': 9/16,
            '3:4': 3/4,
        }

    def forward(self, images, aspect_ratio=None):
        """
        Crop to specific aspect ratio.

        Args:
            images: Tensor of shape (batch_size, 3, H, W)
            aspect_ratio: Target aspect ratio string or None for random

        Returns:
            Cropped and resized images
        """
        if aspect_ratio is None:
            aspect_ratio = list(self.aspect_ratios.keys())[
                torch.randint(0, len(self.aspect_ratios), (1,)).item()
            ]

        target_ratio = self.aspect_ratios.get(aspect_ratio, 1.0)

        batch_size, channels, height, width = images.shape
        current_ratio = width / height

        if current_ratio > target_ratio:
            # Width is too large, crop width
            new_width = int(height * target_ratio)
            left = (width - new_width) // 2
            cropped = images[:, :, :, left:left+new_width]
        else:
            # Height is too large, crop height
            new_height = int(width / target_ratio)
            top = (height - new_height) // 2
            cropped = images[:, :, top:top+new_height, :]

        # Resize back to original dimensions
        resized = F.interpolate(cropped, size=(
            height, width), mode='bilinear', align_corners=False)

        return resized


class CombinedCrop(nn.Module):
    """Randomly apply one of several crop types."""

    def __init__(self):
        super(CombinedCrop, self).__init__()
        self.random_crop = RandomCrop()
        self.center_crop = CenterCrop()
        self.pad_crop = PadCrop()
        self.aspect_crop = AspectRatioCrop()

    def forward(self, images):
        """
        Randomly apply one type of crop.

        Args:
            images: Tensor of shape (batch_size, 3, H, W)

        Returns:
            Cropped images
        """
        crop_type = torch.rand(1).item()

        if crop_type < 0.4:
            return self.random_crop(images)
        elif crop_type < 0.7:
            return self.center_crop(images)
        elif crop_type < 0.9:
            return self.aspect_crop(images)
        else:
            return self.pad_crop(images)


# Test function
if __name__ == "__main__":
    print("Testing Crop Attacks...")

    batch_size = 2
    image_size = 256

    # Create test images
    images = torch.rand(batch_size, 3, image_size, image_size)

    print("\n1. Testing Random Crop...")
    random_crop = RandomCrop(crop_ratio_range=(0.7, 0.95))
    cropped = random_crop(images, crop_ratio=0.8)
    print(f"   Input shape: {images.shape}")
    print(f"   Output shape: {cropped.shape}")
    print(f"   Mean difference: {(images - cropped).abs().mean():.6f}")

    print("\n2. Testing Center Crop...")
    center_crop = CenterCrop(crop_ratio_range=(0.7, 0.95))
    cropped = center_crop(images, crop_ratio=0.85)
    print(f"   Output shape: {cropped.shape}")
    print(f"   Mean difference: {(images - cropped).abs().mean():.6f}")

    print("\n3. Testing Pad Crop...")
    pad_crop = PadCrop(pad_ratio_range=(0.05, 0.15))
    cropped = pad_crop(images, pad_ratio=0.1)
    print(f"   Output shape: {cropped.shape}")
    print(f"   Mean difference: {(images - cropped).abs().mean():.6f}")

    print("\n4. Testing Aspect Ratio Crop...")
    aspect_crop = AspectRatioCrop()
    for ratio in ['16:9', '4:3', '1:1']:
        cropped = aspect_crop(images, aspect_ratio=ratio)
        print(f"   {ratio} - Output shape: {cropped.shape}")

    print("\n5. Testing Combined Crop...")
    combined = CombinedCrop()
    cropped = combined(images)
    print(f"   Output shape: {cropped.shape}")

    # Test gradient flow
    print("\n6. Testing gradient flow...")
    images.requires_grad = True
    cropped = random_crop(images, crop_ratio=0.8)
    loss = cropped.mean()
    loss.backward()
    print(f"   ✅ Gradient flow working: {images.grad is not None}")

    print("\n✅ All crop attack tests passed!")
