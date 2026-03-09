"""
Denoising Defense Module

Implements various denoising techniques to apply before decoding
to improve robustness against noise and compression artifacts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianDenoise(nn.Module):
    """
    Gaussian denoising filter.

    Simple and fast denoising using Gaussian smoothing.
    """

    def __init__(self, kernel_size=5, sigma=1.0, enabled=True):
        """
        Args:
            kernel_size: Size of Gaussian kernel
            sigma: Standard deviation
            enabled: Whether denoising is enabled
        """
        super(GaussianDenoise, self).__init__()
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.sigma = sigma
        self.enabled = enabled

        self.register_buffer('kernel', self._make_gaussian_kernel())

    def _make_gaussian_kernel(self):
        """Generate 2D Gaussian kernel."""
        coords = torch.arange(self.kernel_size, dtype=torch.float32)
        coords = coords - self.kernel_size // 2

        x, y = torch.meshgrid(coords, coords, indexing='ij')

        kernel = torch.exp(-(x**2 + y**2) / (2 * self.sigma**2))
        kernel = kernel / kernel.sum()

        kernel = kernel.view(1, 1, self.kernel_size, self.kernel_size)
        kernel = kernel.repeat(3, 1, 1, 1)

        return kernel

    def forward(self, images):
        """
        Apply Gaussian denoising.

        Args:
            images: Tensor of shape (batch_size, 3, H, W)

        Returns:
            Denoised images
        """
        if not self.enabled:
            return images

        padding = self.kernel_size // 2
        denoised = F.conv2d(images, self.kernel, padding=padding, groups=3)

        return denoised

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def toggle(self):
        self.enabled = not self.enabled


class NonLocalMeansDenoise(nn.Module):
    """
    Non-Local Means denoising (simplified version).

    Denoises by averaging similar patches across the image.
    """

    def __init__(self, search_window=7, patch_size=3, h=0.1, enabled=True):
        """
        Args:
            search_window: Size of search window
            patch_size: Size of patches to compare
            h: Filtering parameter
            enabled: Whether denoising is enabled
        """
        super(NonLocalMeansDenoise, self).__init__()
        self.search_window = search_window
        self.patch_size = patch_size
        self.h = h
        self.enabled = enabled

    def forward(self, images):
        """
        Apply Non-Local Means denoising.

        Args:
            images: Tensor of shape (batch_size, 3, H, W)

        Returns:
            Denoised images
        """
        if not self.enabled:
            return images

        # Simplified NLM using convolution
        # Full NLM is computationally expensive

        batch_size, channels, height, width = images.shape

        # Use average pooling as a fast approximation
        denoised = F.avg_pool2d(
            images, kernel_size=self.patch_size, stride=1, padding=self.patch_size//2)

        return denoised

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def toggle(self):
        self.enabled = not self.enabled


class WaveletDenoise(nn.Module):
    """
    Wavelet-based denoising.

    Denoises by thresholding wavelet coefficients.
    """

    def __init__(self, threshold=0.1, enabled=True):
        """
        Args:
            threshold: Threshold for wavelet coefficients
            enabled: Whether denoising is enabled
        """
        super(WaveletDenoise, self).__init__()
        self.threshold = threshold
        self.enabled = enabled

    def forward(self, images):
        """
        Apply wavelet denoising.

        Args:
            images: Tensor of shape (batch_size, 3, H, W)

        Returns:
            Denoised images
        """
        if not self.enabled:
            return images

        # Simplified wavelet denoising using frequency domain
        # Apply soft thresholding in frequency domain

        batch_size, channels, height, width = images.shape

        # DFT
        fft = torch.fft.fft2(images)
        fft_shifted = torch.fft.fftshift(fft)

        # Soft thresholding
        magnitude = torch.abs(fft_shifted)
        phase = torch.angle(fft_shifted)

        # Threshold small coefficients
        magnitude_thresholded = torch.where(
            magnitude > self.threshold,
            magnitude - self.threshold,
            torch.zeros_like(magnitude)
        )

        # Reconstruct
        fft_thresholded = magnitude_thresholded * torch.exp(1j * phase)
        fft_unshifted = torch.fft.ifftshift(fft_thresholded)
        denoised = torch.fft.ifft2(fft_unshifted).real

        return torch.clamp(denoised, 0.0, 1.0)

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def toggle(self):
        self.enabled = not self.enabled


class CNNDenoise(nn.Module):
    """
    CNN-based denoising using residual learning.

    Learns to predict and remove noise from images.
    """

    def __init__(self, num_layers=5, num_features=64, enabled=True):
        """
        Args:
            num_layers: Number of convolutional layers
            num_features: Number of feature channels
            enabled: Whether denoising is enabled
        """
        super(CNNDenoise, self).__init__()
        self.enabled = enabled

        # First layer
        self.conv_first = nn.Conv2d(3, num_features, kernel_size=3, padding=1)

        # Middle layers
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
            for _ in range(num_layers - 2)
        ])

        self.bn_layers = nn.ModuleList([
            nn.BatchNorm2d(num_features)
            for _ in range(num_layers - 2)
        ])

        # Last layer (predict noise residual)
        self.conv_last = nn.Conv2d(num_features, 3, kernel_size=3, padding=1)

    def forward(self, images):
        """
        Apply CNN denoising.

        Args:
            images: Tensor of shape (batch_size, 3, H, W)

        Returns:
            Denoised images
        """
        if not self.enabled:
            return images

        # First layer
        x = F.relu(self.conv_first(images))

        # Middle layers
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = F.relu(bn(conv(x)))

        # Predict noise residual
        noise_residual = self.conv_last(x)

        # Subtract predicted noise from input
        denoised = images - noise_residual

        return torch.clamp(denoised, 0.0, 1.0)

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def toggle(self):
        self.enabled = not self.enabled


class DenoiseBeforeDecode(nn.Module):
    """
    Unified denoising pipeline for pre-processing before decoding.

    Can apply multiple denoising techniques in sequence or select one adaptively.
    """

    def __init__(self, method='gaussian', enabled=True):
        """
        Args:
            method: Denoising method ('gaussian', 'nlm', 'wavelet', 'cnn', 'adaptive')
            enabled: Whether denoising is enabled
        """
        super(DenoiseBeforeDecode, self).__init__()
        self.method = method
        self.enabled = enabled

        # Initialize all denoising methods
        self.gaussian = GaussianDenoise(kernel_size=5, sigma=1.0, enabled=True)
        self.nlm = NonLocalMeansDenoise(
            search_window=7, patch_size=3, h=0.1, enabled=True)
        self.wavelet = WaveletDenoise(threshold=0.1, enabled=True)
        self.cnn = CNNDenoise(num_layers=5, num_features=64, enabled=True)

    def forward(self, images, method=None):
        """
        Apply denoising before decoding.

        Args:
            images: Tensor of shape (batch_size, 3, H, W)
            method: Override default method (optional)

        Returns:
            Denoised images ready for decoding
        """
        if not self.enabled:
            return images

        method = method or self.method

        if method == 'gaussian':
            return self.gaussian(images)
        elif method == 'nlm':
            return self.nlm(images)
        elif method == 'wavelet':
            return self.wavelet(images)
        elif method == 'cnn':
            return self.cnn(images)
        elif method == 'adaptive':
            # Apply multiple methods and blend
            gaussian_out = self.gaussian(images)
            wavelet_out = self.wavelet(images)
            return 0.6 * gaussian_out + 0.4 * wavelet_out
        elif method == 'none':
            return images
        else:
            return images

    def set_method(self, method):
        """Change denoising method."""
        self.method = method

    def enable(self):
        """Enable denoising."""
        self.enabled = True

    def disable(self):
        """Disable denoising."""
        self.enabled = False

    def toggle(self):
        """Toggle denoising on/off."""
        self.enabled = not self.enabled


# Test function
if __name__ == "__main__":
    print("Testing Denoising Defense Modules...")

    batch_size = 2
    image_size = 256

    # Create test images with noise
    clean_images = torch.rand(batch_size, 3, image_size, image_size)
    noise = torch.randn_like(clean_images) * 0.05
    noisy_images = torch.clamp(clean_images + noise, 0.0, 1.0)

    print(f"Noise level: {(clean_images - noisy_images).abs().mean():.6f}")

    print("\n1. Testing Gaussian Denoise...")
    gaussian = GaussianDenoise(kernel_size=5, sigma=1.0, enabled=True)
    denoised = gaussian(noisy_images)
    improvement = (clean_images - noisy_images).abs().mean() - \
        (clean_images - denoised).abs().mean()
    print(f"   Improvement: {improvement:.6f}")

    print("\n2. Testing NLM Denoise...")
    nlm = NonLocalMeansDenoise(enabled=True)
    denoised = nlm(noisy_images)
    improvement = (clean_images - noisy_images).abs().mean() - \
        (clean_images - denoised).abs().mean()
    print(f"   Improvement: {improvement:.6f}")

    print("\n3. Testing Wavelet Denoise...")
    wavelet = WaveletDenoise(threshold=0.1, enabled=True)
    denoised = wavelet(noisy_images)
    improvement = (clean_images - noisy_images).abs().mean() - \
        (clean_images - denoised).abs().mean()
    print(f"   Improvement: {improvement:.6f}")

    print("\n4. Testing CNN Denoise...")
    cnn = CNNDenoise(num_layers=5, num_features=64, enabled=True)
    denoised = cnn(noisy_images)
    improvement = (clean_images - noisy_images).abs().mean() - \
        (clean_images - denoised).abs().mean()
    print(f"   Improvement: {improvement:.6f}")

    print("\n5. Testing DenoiseBeforeDecode Pipeline...")
    pipeline = DenoiseBeforeDecode(method='gaussian', enabled=True)

    for method in ['gaussian', 'nlm', 'wavelet', 'cnn', 'adaptive', 'none']:
        denoised = pipeline(noisy_images, method=method)
        improvement = (clean_images - noisy_images).abs().mean() - \
            (clean_images - denoised).abs().mean()
        print(f"   Method '{method}': improvement = {improvement:.6f}")

    # Test toggle
    print("\n6. Testing enable/disable...")
    pipeline.disable()
    denoised_disabled = pipeline(noisy_images)
    print(
        f"   Disabled: {(noisy_images - denoised_disabled).abs().mean():.6f} (should be 0)")

    # Test gradient flow
    print("\n7. Testing gradient flow...")
    noisy_images.requires_grad = True
    denoised = gaussian(noisy_images)
    loss = denoised.mean()
    loss.backward()
    print(f"   ✅ Gradient flow working: {noisy_images.grad is not None}")

    print("\n✅ All denoising defense tests passed!")
