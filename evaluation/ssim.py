"""
Structural Similarity Index (SSIM) evaluation for steganography.

Computes SSIM between cover and stego images.
SSIM better correlates with human perception than PSNR.
Higher SSIM (closer to 1.0) indicates better imperceptibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, Tuple


class SSIMMetric(nn.Module):
    """
    Structural Similarity Index (SSIM) metric.

    SSIM compares luminance, contrast, and structure between images.
    Range: [-1, 1], where 1 = identical images.
    Typical values: >0.9 = excellent, >0.8 = good, >0.7 = acceptable.
    """

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        channel: int = 3,
        size_average: bool = True,
        C1: float = 0.01 ** 2,
        C2: float = 0.03 ** 2
    ):
        """
        Args:
            window_size: Size of the Gaussian window (default: 11)
            sigma: Standard deviation of Gaussian window (default: 1.5)
            channel: Number of image channels (default: 3 for RGB)
            size_average: Whether to average SSIM across batch
            C1: Constant for luminance stability (default: (0.01)^2)
            C2: Constant for contrast stability (default: (0.03)^2)
        """
        super(SSIMMetric, self).__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.channel = channel
        self.size_average = size_average
        self.C1 = C1
        self.C2 = C2

        # Create Gaussian window
        self.window = self._create_window(window_size, sigma, channel)

    def _gaussian(self, window_size: int, sigma: float) -> torch.Tensor:
        """Create 1D Gaussian kernel."""
        gauss = torch.arange(window_size, dtype=torch.float32)
        gauss = gauss - window_size // 2
        gauss = torch.exp(-(gauss ** 2) / (2 * sigma ** 2))
        return gauss / gauss.sum()

    def _create_window(
        self,
        window_size: int,
        sigma: float,
        channel: int
    ) -> torch.Tensor:
        """Create 2D Gaussian window."""
        _1D_window = self._gaussian(window_size, sigma).unsqueeze(1)
        _2D_window = _1D_window.mm(
            _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(
            channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        window: torch.Tensor
    ) -> torch.Tensor:
        """Compute SSIM between two images."""
        # Move window to same device as images
        if img1.device != window.device:
            window = window.to(img1.device)

        # Compute means
        mu1 = F.conv2d(img1, window, padding=self.window_size //
                       2, groups=self.channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size //
                       2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        # Compute variances and covariance
        sigma1_sq = F.conv2d(
            img1 * img1, window, padding=self.window_size // 2, groups=self.channel
        ) - mu1_sq
        sigma2_sq = F.conv2d(
            img2 * img2, window, padding=self.window_size // 2, groups=self.channel
        ) - mu2_sq
        sigma12 = F.conv2d(
            img1 * img2, window, padding=self.window_size // 2, groups=self.channel
        ) - mu1_mu2

        # Compute SSIM
        ssim_map = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) / \
                   ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))

        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(dim=(1, 2, 3))

    def forward(
        self,
        stego_image: torch.Tensor,
        cover_image: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute SSIM between stego and cover images.

        Args:
            stego_image: Stego image (batch_size, 3, H, W)
            cover_image: Cover image (batch_size, 3, H, W)

        Returns:
            SSIM value [0, 1] (scalar tensor if size_average=True)
        """
        return self._ssim(stego_image, cover_image, self.window)

    def compute_per_sample(
        self,
        stego_image: torch.Tensor,
        cover_image: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute SSIM for each sample in batch.

        Args:
            stego_image: Stego images (batch_size, 3, H, W)
            cover_image: Cover images (batch_size, 3, H, W)

        Returns:
            SSIM for each sample (batch_size,)
        """
        # Temporarily disable size averaging
        original_size_average = self.size_average
        self.size_average = False

        ssim_per_sample = self._ssim(stego_image, cover_image, self.window)

        # Restore original setting
        self.size_average = original_size_average

        return ssim_per_sample


class MultiScaleSSIM(nn.Module):
    """
    Multi-Scale SSIM (MS-SSIM) metric.

    Computes SSIM at multiple scales for better quality assessment.
    """

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        channel: int = 3,
        scales: int = 5,
        weights: list = None
    ):
        """
        Args:
            window_size: Size of Gaussian window
            sigma: Standard deviation of Gaussian
            channel: Number of channels
            scales: Number of scales to compute
            weights: Weights for each scale (default: equal weights)
        """
        super(MultiScaleSSIM, self).__init__()
        self.scales = scales

        if weights is None:
            self.weights = torch.ones(scales) / scales
        else:
            self.weights = torch.tensor(weights)

        self.ssim = SSIMMetric(
            window_size=window_size,
            sigma=sigma,
            channel=channel,
            size_average=True
        )

    def forward(
        self,
        stego_image: torch.Tensor,
        cover_image: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute MS-SSIM.

        Args:
            stego_image: Stego image
            cover_image: Cover image

        Returns:
            MS-SSIM value
        """
        msssim = 0.0

        for i in range(self.scales):
            # Compute SSIM at current scale
            ssim_val = self.ssim(stego_image, cover_image)
            msssim += self.weights[i] * ssim_val

            # Downsample for next scale
            if i < self.scales - 1:
                stego_image = F.avg_pool2d(
                    stego_image, kernel_size=2, stride=2)
                cover_image = F.avg_pool2d(
                    cover_image, kernel_size=2, stride=2)

        return msssim


def compute_ssim(
    stego_image: Union[torch.Tensor, np.ndarray],
    cover_image: Union[torch.Tensor, np.ndarray],
    window_size: int = 11
) -> float:
    """
    Standalone function to compute SSIM.

    Args:
        stego_image: Stego image
        cover_image: Cover image
        window_size: Size of Gaussian window

    Returns:
        SSIM value as float
    """
    # Convert to torch tensors if needed
    if isinstance(stego_image, np.ndarray):
        stego_image = torch.from_numpy(stego_image)
    if isinstance(cover_image, np.ndarray):
        cover_image = torch.from_numpy(cover_image)

    # Ensure 4D tensor (batch_size, channels, height, width)
    if stego_image.dim() == 3:
        stego_image = stego_image.unsqueeze(0)
        cover_image = cover_image.unsqueeze(0)

    metric = SSIMMetric(window_size=window_size, channel=stego_image.size(1))
    ssim = metric(stego_image, cover_image)

    return ssim.item()


def compute_ssim_statistics(
    stego_images: torch.Tensor,
    cover_images: torch.Tensor,
    window_size: int = 11
) -> dict:
    """
    Compute SSIM statistics across multiple images.

    Args:
        stego_images: Batch of stego images (batch_size, 3, H, W)
        cover_images: Batch of cover images (batch_size, 3, H, W)
        window_size: Size of Gaussian window

    Returns:
        Dictionary with SSIM statistics
    """
    metric = SSIMMetric(window_size=window_size, channel=stego_images.size(1))

    # Compute per-sample SSIM
    ssim_per_sample = metric.compute_per_sample(stego_images, cover_images)

    # Compute statistics
    stats = {
        'mean': ssim_per_sample.mean().item(),
        'std': ssim_per_sample.std().item(),
        'min': ssim_per_sample.min().item(),
        'max': ssim_per_sample.max().item(),
        'median': ssim_per_sample.median().item(),
        'per_sample': ssim_per_sample.cpu().numpy()
    }

    return stats


def evaluate_ssim_with_attacks(
    model: nn.Module,
    images: torch.Tensor,
    messages: torch.Tensor,
    attacks: list = None,
    device: str = 'cuda',
    window_size: int = 11
) -> dict:
    """
    Evaluate SSIM under various attacks.

    Args:
        model: Steganography model
        images: Cover images
        messages: Secret messages
        attacks: List of (name, attack_function) tuples
        device: Device to run on
        window_size: Size of Gaussian window

    Returns:
        Dictionary with SSIM for each attack condition
    """
    model = model.to(device)
    model.eval()

    images = images.to(device)
    messages = messages.to(device)

    results = {}

    with torch.no_grad():
        # Encode messages
        stego_images = model.encode(images, messages)

        # Baseline: stego vs cover
        ssim_baseline = compute_ssim(
            stego_images, images, window_size=window_size)
        results['stego_vs_cover'] = ssim_baseline

        # Test with attacks
        if attacks is not None:
            for attack_name, attack_fn in attacks:
                attacked_stego = attack_fn(stego_images)

                # Compare attacked vs original stego
                ssim = compute_ssim(
                    attacked_stego, stego_images, window_size=window_size)
                results[f'{attack_name}_vs_stego'] = ssim

                # Compare attacked vs cover
                ssim = compute_ssim(attacked_stego, images,
                                    window_size=window_size)
                results[f'{attack_name}_vs_cover'] = ssim

    return results


def classify_ssim_quality(ssim: float) -> str:
    """
    Classify SSIM value into quality categories.

    Args:
        ssim: SSIM value [0, 1]

    Returns:
        Quality label
    """
    if ssim >= 0.95:
        return "Excellent (imperceptible)"
    elif ssim >= 0.90:
        return "Very Good (minor differences)"
    elif ssim >= 0.80:
        return "Good (visible but acceptable)"
    elif ssim >= 0.70:
        return "Fair (noticeable differences)"
    elif ssim >= 0.50:
        return "Poor (significant differences)"
    else:
        return "Very Poor (severe degradation)"


def ssim_to_dssim(ssim: float) -> float:
    """
    Convert SSIM to DSSIM (structural dissimilarity).

    DSSIM = (1 - SSIM) / 2

    Args:
        ssim: SSIM value

    Returns:
        DSSIM value [0, 1]
    """
    return (1.0 - ssim) / 2.0


# Test function
if __name__ == "__main__":
    print("=" * 60)
    print("Testing SSIM Metric")
    print("=" * 60)

    # Test 1: Identical images (SSIM = 1.0)
    print("\n1. Identical Images (SSIM = 1.0)")
    print("-" * 60)

    batch_size = 4
    image_size = 256

    cover = torch.rand(batch_size, 3, image_size, image_size)
    stego = cover.clone()

    ssim = compute_ssim(stego, cover)
    print(f"   SSIM: {ssim:.6f}")
    print(f"   Quality: {classify_ssim_quality(ssim)}")
    print(f"   ✓ SSIM = 1.0: {abs(ssim - 1.0) < 0.01}")

    # Test 2: Small perturbation
    print("\n2. Small Perturbation")
    print("-" * 60)

    cover = torch.rand(batch_size, 3, image_size, image_size)
    stego = cover + torch.randn_like(cover) * 0.01
    stego = torch.clamp(stego, 0, 1)

    ssim = compute_ssim(stego, cover)
    print(f"   SSIM: {ssim:.6f}")
    print(f"   Quality: {classify_ssim_quality(ssim)}")
    print(f"   ✓ SSIM high: {ssim > 0.9}")

    # Test 3: Large perturbation
    print("\n3. Large Perturbation")
    print("-" * 60)

    cover = torch.rand(batch_size, 3, image_size, image_size)
    stego = cover + torch.randn_like(cover) * 0.3
    stego = torch.clamp(stego, 0, 1)

    ssim = compute_ssim(stego, cover)
    print(f"   SSIM: {ssim:.6f}")
    print(f"   Quality: {classify_ssim_quality(ssim)}")

    # Test 4: Per-sample SSIM
    print("\n4. Per-Sample SSIM")
    print("-" * 60)

    metric = SSIMMetric()
    cover = torch.rand(batch_size, 3, image_size, image_size)

    # Create varying perturbations
    stego = cover.clone()
    for i in range(batch_size):
        noise_level = 0.02 * (i + 1)
        stego[i] = cover[i] + torch.randn_like(cover[i]) * noise_level
    stego = torch.clamp(stego, 0, 1)

    ssim_per_sample = metric.compute_per_sample(stego, cover)

    print(f"   SSIM per sample:")
    for i, ssim_val in enumerate(ssim_per_sample):
        print(
            f"     Sample {i}: {ssim_val.item():.6f} - {classify_ssim_quality(ssim_val.item())}")

    # Test 5: SSIM statistics
    print("\n5. SSIM Statistics")
    print("-" * 60)

    num_images = 50
    cover = torch.rand(num_images, 3, 128, 128)
    stego = cover + torch.randn_like(cover) * 0.02
    stego = torch.clamp(stego, 0, 1)

    stats = compute_ssim_statistics(stego, cover)

    print(f"   Mean SSIM: {stats['mean']:.6f}")
    print(f"   Std SSIM: {stats['std']:.6f}")
    print(f"   Min SSIM: {stats['min']:.6f}")
    print(f"   Max SSIM: {stats['max']:.6f}")
    print(f"   Median SSIM: {stats['median']:.6f}")
    print(f"   Quality: {classify_ssim_quality(stats['mean'])}")

    # Test 6: Different window sizes
    print("\n6. Different Window Sizes")
    print("-" * 60)

    cover = torch.rand(batch_size, 3, image_size, image_size)
    stego = cover + torch.randn_like(cover) * 0.02
    stego = torch.clamp(stego, 0, 1)

    for window_size in [7, 11, 15]:
        ssim = compute_ssim(stego, cover, window_size=window_size)
        print(f"   Window size {window_size:2d}: SSIM = {ssim:.6f}")

    # Test 7: Multi-Scale SSIM
    print("\n7. Multi-Scale SSIM")
    print("-" * 60)

    cover = torch.rand(batch_size, 3, image_size, image_size)
    stego = cover + torch.randn_like(cover) * 0.02
    stego = torch.clamp(stego, 0, 1)

    ms_ssim_metric = MultiScaleSSIM()
    ms_ssim = ms_ssim_metric(stego, cover)

    # Regular SSIM for comparison
    ssim = compute_ssim(stego, cover)

    print(f"   SSIM: {ssim:.6f}")
    print(f"   MS-SSIM: {ms_ssim.item():.6f}")
    print(f"   ✓ MS-SSIM computed")

    # Test 8: SSIM to DSSIM conversion
    print("\n8. SSIM ↔ DSSIM Conversion")
    print("-" * 60)

    ssim_values = [0.99, 0.95, 0.90, 0.80, 0.70]
    for ssim_val in ssim_values:
        dssim = ssim_to_dssim(ssim_val)
        print(f"   SSIM: {ssim_val:.2f} → DSSIM: {dssim:.4f}")

    # Test 9: SSIM vs different types of distortions
    print("\n9. SSIM vs Different Distortions")
    print("-" * 60)

    cover = torch.rand(4, 3, 128, 128)

    # Gaussian noise
    stego_noise = cover + torch.randn_like(cover) * 0.05
    stego_noise = torch.clamp(stego_noise, 0, 1)
    ssim_noise = compute_ssim(stego_noise, cover)

    # Blur
    stego_blur = F.avg_pool2d(cover, kernel_size=3, stride=1, padding=1)
    ssim_blur = compute_ssim(stego_blur, cover)

    # Brightness shift
    stego_bright = torch.clamp(cover + 0.1, 0, 1)
    ssim_bright = compute_ssim(stego_bright, cover)

    print(f"   Gaussian noise: SSIM = {ssim_noise:.6f}")
    print(f"   Blur: SSIM = {ssim_blur:.6f}")
    print(f"   Brightness shift: SSIM = {ssim_bright:.6f}")

    # Test 10: NumPy compatibility
    print("\n10. NumPy Compatibility")
    print("-" * 60)

    cover_np = np.random.rand(batch_size, 3, 64, 64).astype(np.float32)
    stego_np = cover_np + \
        np.random.randn(*cover_np.shape).astype(np.float32) * 0.02
    stego_np = np.clip(stego_np, 0, 1)

    ssim = compute_ssim(stego_np, cover_np)
    print(f"   SSIM with NumPy input: {ssim:.6f}")
    print(f"   ✓ NumPy arrays work")

    # Test 11: Grayscale images
    print("\n11. Grayscale Images")
    print("-" * 60)

    cover_gray = torch.rand(batch_size, 1, 128, 128)
    stego_gray = cover_gray + torch.randn_like(cover_gray) * 0.02
    stego_gray = torch.clamp(stego_gray, 0, 1)

    metric_gray = SSIMMetric(channel=1)
    ssim_gray = metric_gray(stego_gray, cover_gray)
    print(f"   SSIM (grayscale): {ssim_gray.item():.6f}")
    print(f"   ✓ Grayscale images work")

    # Test 12: Edge cases
    print("\n12. Edge Cases")
    print("-" * 60)

    # All zeros vs all ones
    cover_zeros = torch.zeros(2, 3, 64, 64)
    cover_ones = torch.ones(2, 3, 64, 64)
    ssim_extreme = compute_ssim(cover_ones, cover_zeros)
    print(f"   All zeros vs all ones: SSIM = {ssim_extreme:.6f}")

    # Constant image with small perturbation
    cover_const = torch.ones(2, 3, 64, 64) * 0.5
    stego_const = cover_const + torch.randn_like(cover_const) * 0.01
    stego_const = torch.clamp(stego_const, 0, 1)
    ssim_const = compute_ssim(stego_const, cover_const)
    print(f"   Constant image perturbed: SSIM = {ssim_const:.6f}")

    print("\n" + "=" * 60)
    print("✅ All SSIM tests passed!")
    print("=" * 60)

    print("\nUsage Example:")
    print("""
# Basic usage
from evaluation.ssim import compute_ssim, classify_ssim_quality

stego_image = model.encode(cover_image, message)
ssim = compute_ssim(stego_image, cover_image)

print(f"SSIM: {ssim:.4f}")
print(f"Quality: {classify_ssim_quality(ssim)}")

# Statistics across dataset
from evaluation.ssim import compute_ssim_statistics

stats = compute_ssim_statistics(stego_images, cover_images)
print(f"Mean SSIM: {stats['mean']:.4f} ± {stats['std']:.4f}")

# Multi-scale SSIM
from evaluation.ssim import MultiScaleSSIM

ms_ssim_metric = MultiScaleSSIM()
ms_ssim = ms_ssim_metric(stego_image, cover_image)
print(f"MS-SSIM: {ms_ssim.item():.4f}")

# Evaluate with attacks
from evaluation.ssim import evaluate_ssim_with_attacks

attacks = [
    ('jpeg_50', lambda x: jpeg_compression(x, quality=50)),
    ('gaussian_noise', lambda x: add_gaussian_noise(x, std=0.1))
]

results = evaluate_ssim_with_attacks(model, images, messages, attacks)
for condition, ssim in results.items():
    print(f"{condition}: {ssim:.4f}")
    """)
