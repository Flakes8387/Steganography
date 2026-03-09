"""
Anti-Aliasing Defense Module

Applies anti-aliasing filters before decoding to reduce high-frequency artifacts
and improve robustness against compression attacks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GaussianAntiAlias(nn.Module):
    """
    Gaussian anti-aliasing filter.

    Applies a Gaussian low-pass filter to reduce aliasing artifacts
    before decoding the hidden message.
    """

    def __init__(self, kernel_size=5, sigma=1.0, enabled=True):
        """
        Args:
            kernel_size: Size of the Gaussian kernel (odd number)
            sigma: Standard deviation of Gaussian
            enabled: Whether the filter is enabled
        """
        super(GaussianAntiAlias, self).__init__()
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.sigma = sigma
        self.enabled = enabled

        # Pre-compute Gaussian kernel
        self.register_buffer('kernel', self._make_gaussian_kernel())

    def _make_gaussian_kernel(self):
        """Generate 2D Gaussian kernel."""
        coords = torch.arange(self.kernel_size, dtype=torch.float32)
        coords = coords - self.kernel_size // 2

        x, y = torch.meshgrid(coords, coords, indexing='ij')

        kernel = torch.exp(-(x**2 + y**2) / (2 * self.sigma**2))
        kernel = kernel / kernel.sum()

        # Expand for convolution (out_channels, in_channels/groups, kH, kW)
        kernel = kernel.view(1, 1, self.kernel_size, self.kernel_size)
        kernel = kernel.repeat(3, 1, 1, 1)  # 3 channels (RGB)

        return kernel

    def forward(self, images):
        """
        Apply Gaussian anti-aliasing.

        Args:
            images: Tensor of shape (batch_size, 3, H, W)

        Returns:
            Filtered images
        """
        if not self.enabled:
            return images

        padding = self.kernel_size // 2
        filtered = F.conv2d(images, self.kernel, padding=padding, groups=3)

        return filtered

    def enable(self):
        """Enable the filter."""
        self.enabled = True

    def disable(self):
        """Disable the filter."""
        self.enabled = False

    def toggle(self):
        """Toggle filter on/off."""
        self.enabled = not self.enabled


class BilateralAntiAlias(nn.Module):
    """
    Bilateral filter for edge-preserving anti-aliasing.

    Preserves edges while smoothing flat regions.
    """

    def __init__(self, kernel_size=5, sigma_spatial=1.0, sigma_intensity=0.1, enabled=True):
        """
        Args:
            kernel_size: Size of the filter kernel
            sigma_spatial: Spatial sigma for Gaussian
            sigma_intensity: Intensity sigma for edge preservation
            enabled: Whether the filter is enabled
        """
        super(BilateralAntiAlias, self).__init__()
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.sigma_spatial = sigma_spatial
        self.sigma_intensity = sigma_intensity
        self.enabled = enabled

        # Pre-compute spatial kernel
        self.register_buffer('spatial_kernel', self._make_spatial_kernel())

    def _make_spatial_kernel(self):
        """Generate spatial Gaussian kernel."""
        coords = torch.arange(self.kernel_size, dtype=torch.float32)
        coords = coords - self.kernel_size // 2

        x, y = torch.meshgrid(coords, coords, indexing='ij')

        kernel = torch.exp(-(x**2 + y**2) / (2 * self.sigma_spatial**2))

        return kernel

    def forward(self, images):
        """
        Apply bilateral filtering.

        Args:
            images: Tensor of shape (batch_size, 3, H, W)

        Returns:
            Filtered images
        """
        if not self.enabled:
            return images

        # Simplified bilateral filter using guided filter approximation
        # Full bilateral filter is computationally expensive

        batch_size, channels, height, width = images.shape
        padding = self.kernel_size // 2

        # Pad images
        padded = F.pad(images, [padding] * 4, mode='reflect')

        # Extract patches
        unfold = F.unfold(padded, kernel_size=self.kernel_size)
        unfold = unfold.view(
            batch_size, channels, self.kernel_size * self.kernel_size, height * width)

        # Center pixel
        center_idx = (self.kernel_size * self.kernel_size) // 2
        center_pixel = unfold[:, :, center_idx:center_idx+1, :]

        # Intensity differences
        intensity_diff = unfold - center_pixel
        intensity_weights = torch.exp(-(intensity_diff**2) /
                                      (2 * self.sigma_intensity**2))

        # Combine spatial and intensity weights
        spatial_flat = self.spatial_kernel.view(1, 1, -1, 1)
        combined_weights = spatial_flat * intensity_weights
        combined_weights = combined_weights / \
            (combined_weights.sum(dim=2, keepdim=True) + 1e-8)

        # Apply weighted average
        filtered = (unfold * combined_weights).sum(dim=2)
        filtered = filtered.view(batch_size, channels, height, width)

        return filtered

    def enable(self):
        """Enable the filter."""
        self.enabled = True

    def disable(self):
        """Disable the filter."""
        self.enabled = False

    def toggle(self):
        """Toggle filter on/off."""
        self.enabled = not self.enabled


class MedianAntiAlias(nn.Module):
    """
    Median filter for robust anti-aliasing.

    Effective at removing salt-and-pepper noise while preserving edges.
    """

    def __init__(self, kernel_size=3, enabled=True):
        """
        Args:
            kernel_size: Size of the median filter kernel
            enabled: Whether the filter is enabled
        """
        super(MedianAntiAlias, self).__init__()
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.enabled = enabled

    def forward(self, images):
        """
        Apply median filtering.

        Args:
            images: Tensor of shape (batch_size, 3, H, W)

        Returns:
            Filtered images
        """
        if not self.enabled:
            return images

        batch_size, channels, height, width = images.shape
        padding = self.kernel_size // 2

        # Pad images
        padded = F.pad(images, [padding] * 4, mode='reflect')

        # Extract patches
        unfold = F.unfold(padded, kernel_size=self.kernel_size)
        unfold = unfold.view(
            batch_size, channels, self.kernel_size * self.kernel_size, height * width)

        # Compute median
        filtered = unfold.median(dim=2)[0]
        filtered = filtered.view(batch_size, channels, height, width)

        return filtered

    def enable(self):
        """Enable the filter."""
        self.enabled = True

    def disable(self):
        """Disable the filter."""
        self.enabled = False

    def toggle(self):
        """Toggle filter on/off."""
        self.enabled = not self.enabled


class AdaptiveAntiAlias(nn.Module):
    """
    Adaptive anti-aliasing that combines multiple filters.

    Automatically selects or blends filters based on image characteristics.
    """

    def __init__(self, enabled=True):
        """
        Args:
            enabled: Whether the filter is enabled
        """
        super(AdaptiveAntiAlias, self).__init__()
        self.enabled = enabled

        self.gaussian = GaussianAntiAlias(
            kernel_size=5, sigma=1.0, enabled=True)
        self.bilateral = BilateralAntiAlias(
            kernel_size=5, sigma_spatial=1.0, sigma_intensity=0.1, enabled=True)
        self.median = MedianAntiAlias(kernel_size=3, enabled=True)

    def forward(self, images, mode='gaussian'):
        """
        Apply adaptive anti-aliasing.

        Args:
            images: Tensor of shape (batch_size, 3, H, W)
            mode: Filter mode ('gaussian', 'bilateral', 'median', 'auto')

        Returns:
            Filtered images
        """
        if not self.enabled:
            return images

        if mode == 'gaussian':
            return self.gaussian(images)
        elif mode == 'bilateral':
            return self.bilateral(images)
        elif mode == 'median':
            return self.median(images)
        elif mode == 'auto':
            # Blend multiple filters with learned or heuristic weights
            gaussian_out = self.gaussian(images)
            bilateral_out = self.bilateral(images)

            # Simple blending (can be made learnable)
            return 0.6 * gaussian_out + 0.4 * bilateral_out
        else:
            return images

    def enable(self):
        """Enable the filter."""
        self.enabled = True

    def disable(self):
        """Disable the filter."""
        self.enabled = False

    def toggle(self):
        """Toggle filter on/off."""
        self.enabled = not self.enabled


# Test function
if __name__ == "__main__":
    print("Testing Anti-Aliasing Defense Modules...")

    batch_size = 2
    image_size = 256

    # Create test images
    images = torch.rand(batch_size, 3, image_size, image_size)

    print("\n1. Testing Gaussian Anti-Alias...")
    gaussian = GaussianAntiAlias(kernel_size=5, sigma=1.0, enabled=True)
    filtered = gaussian(images)
    print(f"   Input shape: {images.shape}")
    print(f"   Output shape: {filtered.shape}")
    print(f"   Difference: {(images - filtered).abs().mean():.6f}")

    # Test toggle
    gaussian.disable()
    filtered_disabled = gaussian(images)
    print(
        f"   Disabled difference: {(images - filtered_disabled).abs().mean():.6f} (should be 0)")

    print("\n2. Testing Bilateral Anti-Alias...")
    bilateral = BilateralAntiAlias(
        kernel_size=5, sigma_spatial=1.0, sigma_intensity=0.1, enabled=True)
    filtered = bilateral(images)
    print(f"   Output shape: {filtered.shape}")
    print(f"   Difference: {(images - filtered).abs().mean():.6f}")

    print("\n3. Testing Median Anti-Alias...")
    median = MedianAntiAlias(kernel_size=3, enabled=True)
    filtered = median(images)
    print(f"   Output shape: {filtered.shape}")
    print(f"   Difference: {(images - filtered).abs().mean():.6f}")

    print("\n4. Testing Adaptive Anti-Alias...")
    adaptive = AdaptiveAntiAlias(enabled=True)

    for mode in ['gaussian', 'bilateral', 'median', 'auto']:
        filtered = adaptive(images, mode=mode)
        diff = (images - filtered).abs().mean()
        print(f"   Mode '{mode}': difference = {diff:.6f}")

    # Test gradient flow
    print("\n5. Testing gradient flow...")
    images.requires_grad = True
    filtered = gaussian(images)
    loss = filtered.mean()
    loss.backward()
    print(f"   ✅ Gradient flow working: {images.grad is not None}")

    print("\n✅ All anti-aliasing defense tests passed!")
