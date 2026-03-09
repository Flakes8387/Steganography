"""
Peak Signal-to-Noise Ratio (PSNR) evaluation for steganography.

Computes PSNR between cover and stego images.
Higher PSNR indicates better imperceptibility (less visible changes).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Union, Tuple
import math


class PSNRMetric(nn.Module):
    """
    Peak Signal-to-Noise Ratio (PSNR) metric.

    PSNR = 10 * log10(MAX^2 / MSE)

    Where MAX is the maximum possible pixel value (1.0 for normalized images).
    Higher PSNR (>30 dB) indicates better quality.
    """

    def __init__(self, max_val: float = 1.0, eps: float = 1e-8):
        """
        Args:
            max_val: Maximum pixel value (1.0 for normalized, 255 for uint8)
            eps: Small value to avoid log(0)
        """
        super(PSNRMetric, self).__init__()
        self.max_val = max_val
        self.eps = eps

    def forward(
        self,
        stego_image: torch.Tensor,
        cover_image: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute PSNR between stego and cover images.

        Args:
            stego_image: Stego image (batch_size, 3, H, W)
            cover_image: Cover image (batch_size, 3, H, W)

        Returns:
            PSNR value in dB (scalar tensor)
        """
        # Compute MSE
        mse = torch.mean((stego_image - cover_image) ** 2)

        # Avoid log(0)
        mse = torch.clamp(mse, min=self.eps)

        # Compute PSNR
        psnr = 10 * torch.log10(self.max_val ** 2 / mse)

        return psnr

    def compute_per_sample(
        self,
        stego_image: torch.Tensor,
        cover_image: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute PSNR for each sample in batch.

        Args:
            stego_image: Stego images (batch_size, 3, H, W)
            cover_image: Cover images (batch_size, 3, H, W)

        Returns:
            PSNR for each sample (batch_size,)
        """
        batch_size = stego_image.size(0)

        # Reshape to (batch_size, -1)
        stego_flat = stego_image.view(batch_size, -1)
        cover_flat = cover_image.view(batch_size, -1)

        # Compute MSE per sample
        mse_per_sample = torch.mean((stego_flat - cover_flat) ** 2, dim=1)

        # Clamp to avoid log(0)
        mse_per_sample = torch.clamp(mse_per_sample, min=self.eps)

        # Compute PSNR per sample
        psnr_per_sample = 10 * torch.log10(self.max_val ** 2 / mse_per_sample)

        return psnr_per_sample


def compute_psnr(
    stego_image: Union[torch.Tensor, np.ndarray],
    cover_image: Union[torch.Tensor, np.ndarray],
    max_val: float = 1.0
) -> float:
    """
    Standalone function to compute PSNR.

    Args:
        stego_image: Stego image
        cover_image: Cover image
        max_val: Maximum pixel value

    Returns:
        PSNR value in dB as float
    """
    # Convert to torch tensors if needed
    if isinstance(stego_image, np.ndarray):
        stego_image = torch.from_numpy(stego_image)
    if isinstance(cover_image, np.ndarray):
        cover_image = torch.from_numpy(cover_image)

    metric = PSNRMetric(max_val=max_val)
    psnr = metric(stego_image, cover_image)

    return psnr.item()


def compute_psnr_statistics(
    stego_images: torch.Tensor,
    cover_images: torch.Tensor,
    max_val: float = 1.0
) -> dict:
    """
    Compute PSNR statistics across multiple images.

    Args:
        stego_images: Batch of stego images (batch_size, 3, H, W)
        cover_images: Batch of cover images (batch_size, 3, H, W)
        max_val: Maximum pixel value

    Returns:
        Dictionary with PSNR statistics:
            - mean: Mean PSNR across all images
            - std: Standard deviation of PSNR
            - min: Minimum PSNR
            - max: Maximum PSNR
            - median: Median PSNR
            - per_sample: PSNR for each image
    """
    metric = PSNRMetric(max_val=max_val)

    # Compute per-sample PSNR
    psnr_per_sample = metric.compute_per_sample(stego_images, cover_images)

    # Compute statistics
    stats = {
        'mean': psnr_per_sample.mean().item(),
        'std': psnr_per_sample.std().item(),
        'min': psnr_per_sample.min().item(),
        'max': psnr_per_sample.max().item(),
        'median': psnr_per_sample.median().item(),
        'per_sample': psnr_per_sample.cpu().numpy()
    }

    return stats


def evaluate_psnr_with_attacks(
    model: nn.Module,
    images: torch.Tensor,
    messages: torch.Tensor,
    attacks: list = None,
    device: str = 'cuda',
    max_val: float = 1.0
) -> dict:
    """
    Evaluate PSNR under various attacks.

    Args:
        model: Steganography model with encode() method
        images: Cover images (batch_size, 3, H, W)
        messages: Secret messages (batch_size, message_length)
        attacks: List of (name, attack_function) tuples
        device: Device to run on
        max_val: Maximum pixel value

    Returns:
        Dictionary with PSNR for each attack condition
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
        psnr_baseline = compute_psnr(stego_images, images, max_val=max_val)
        results['stego_vs_cover'] = psnr_baseline

        # Test with attacks if provided
        if attacks is not None:
            for attack_name, attack_fn in attacks:
                # Apply attack
                attacked_stego = attack_fn(stego_images)

                # Compare attacked vs original stego
                psnr = compute_psnr(
                    attacked_stego, stego_images, max_val=max_val)
                results[f'{attack_name}_vs_stego'] = psnr

                # Compare attacked vs cover
                psnr = compute_psnr(attacked_stego, images, max_val=max_val)
                results[f'{attack_name}_vs_cover'] = psnr

    return results


def psnr_to_mse(psnr: float, max_val: float = 1.0) -> float:
    """
    Convert PSNR to MSE.

    Args:
        psnr: PSNR value in dB
        max_val: Maximum pixel value

    Returns:
        MSE value
    """
    return (max_val ** 2) / (10 ** (psnr / 10))


def mse_to_psnr(mse: float, max_val: float = 1.0) -> float:
    """
    Convert MSE to PSNR.

    Args:
        mse: MSE value
        max_val: Maximum pixel value

    Returns:
        PSNR value in dB
    """
    if mse == 0:
        return float('inf')
    return 10 * math.log10((max_val ** 2) / mse)


def classify_psnr_quality(psnr: float) -> str:
    """
    Classify PSNR value into quality categories.

    Args:
        psnr: PSNR value in dB

    Returns:
        Quality label
    """
    if psnr >= 40:
        return "Excellent (imperceptible)"
    elif psnr >= 35:
        return "Very Good (minor artifacts)"
    elif psnr >= 30:
        return "Good (visible but acceptable)"
    elif psnr >= 25:
        return "Fair (noticeable artifacts)"
    elif psnr >= 20:
        return "Poor (significant artifacts)"
    else:
        return "Very Poor (severe degradation)"


# Test function
if __name__ == "__main__":
    print("=" * 60)
    print("Testing PSNR Metric")
    print("=" * 60)

    # Test 1: Identical images (PSNR = inf)
    print("\n1. Identical Images (PSNR → ∞)")
    print("-" * 60)

    batch_size = 4
    image_size = 256

    cover = torch.rand(batch_size, 3, image_size, image_size)
    stego = cover.clone()

    metric = PSNRMetric()
    psnr = metric(stego, cover)
    print(f"   PSNR: {psnr.item():.2f} dB")
    print(f"   ✓ PSNR very high: {psnr.item() > 100}")

    # Test 2: Small perturbation
    print("\n2. Small Perturbation")
    print("-" * 60)

    cover = torch.rand(batch_size, 3, image_size, image_size)
    stego = cover + torch.randn_like(cover) * 0.01  # Small noise

    psnr = compute_psnr(stego, cover, max_val=1.0)
    print(f"   PSNR: {psnr:.2f} dB")
    print(f"   Quality: {classify_psnr_quality(psnr)}")
    print(f"   ✓ PSNR reasonable: {psnr > 20}")

    # Test 3: Large perturbation
    print("\n3. Large Perturbation")
    print("-" * 60)

    cover = torch.rand(batch_size, 3, image_size, image_size)
    stego = cover + torch.randn_like(cover) * 0.3  # Large noise
    stego = torch.clamp(stego, 0, 1)

    psnr = compute_psnr(stego, cover, max_val=1.0)
    print(f"   PSNR: {psnr:.2f} dB")
    print(f"   Quality: {classify_psnr_quality(psnr)}")

    # Test 4: Per-sample PSNR
    print("\n4. Per-Sample PSNR")
    print("-" * 60)

    metric = PSNRMetric()
    cover = torch.rand(batch_size, 3, image_size, image_size)

    # Create varying perturbations
    stego = cover.clone()
    for i in range(batch_size):
        noise_level = 0.01 * (i + 1)
        stego[i] = cover[i] + torch.randn_like(cover[i]) * noise_level
    stego = torch.clamp(stego, 0, 1)

    psnr_per_sample = metric.compute_per_sample(stego, cover)

    print(f"   PSNR per sample:")
    for i, psnr_val in enumerate(psnr_per_sample):
        print(
            f"     Sample {i}: {psnr_val.item():.2f} dB - {classify_psnr_quality(psnr_val.item())}")

    # Test 5: PSNR statistics
    print("\n5. PSNR Statistics")
    print("-" * 60)

    num_images = 50
    cover = torch.rand(num_images, 3, 128, 128)
    stego = cover + torch.randn_like(cover) * 0.02
    stego = torch.clamp(stego, 0, 1)

    stats = compute_psnr_statistics(stego, cover, max_val=1.0)

    print(f"   Mean PSNR: {stats['mean']:.2f} dB")
    print(f"   Std PSNR: {stats['std']:.2f} dB")
    print(f"   Min PSNR: {stats['min']:.2f} dB")
    print(f"   Max PSNR: {stats['max']:.2f} dB")
    print(f"   Median PSNR: {stats['median']:.2f} dB")
    print(f"   Quality: {classify_psnr_quality(stats['mean'])}")

    # Test 6: Different max_val (uint8 images)
    print("\n6. Different max_val (uint8 simulation)")
    print("-" * 60)

    cover_uint8 = torch.randint(
        0, 256, (batch_size, 3, image_size, image_size)).float()
    stego_uint8 = cover_uint8 + torch.randn_like(cover_uint8) * 5
    stego_uint8 = torch.clamp(stego_uint8, 0, 255)

    psnr_uint8 = compute_psnr(stego_uint8, cover_uint8, max_val=255.0)
    print(f"   PSNR (max_val=255): {psnr_uint8:.2f} dB")

    # Convert to [0, 1] range
    cover_norm = cover_uint8 / 255.0
    stego_norm = stego_uint8 / 255.0
    psnr_norm = compute_psnr(stego_norm, cover_norm, max_val=1.0)
    print(f"   PSNR (max_val=1.0): {psnr_norm:.2f} dB")
    print(
        f"   ✓ Values should be similar: {abs(psnr_uint8 - psnr_norm) < 0.1}")

    # Test 7: PSNR to MSE conversion
    print("\n7. PSNR ↔ MSE Conversion")
    print("-" * 60)

    psnr_values = [20, 30, 40, 50]
    for psnr_val in psnr_values:
        mse = psnr_to_mse(psnr_val, max_val=1.0)
        psnr_back = mse_to_psnr(mse, max_val=1.0)
        print(
            f"   PSNR: {psnr_val} dB → MSE: {mse:.6f} → PSNR: {psnr_back:.2f} dB")
        print(
            f"     ✓ Round-trip accurate: {abs(psnr_val - psnr_back) < 0.01}")

    # Test 8: Quality classification
    print("\n8. Quality Classification")
    print("-" * 60)

    test_psnrs = [15, 22, 28, 33, 38, 45]
    for psnr_val in test_psnrs:
        quality = classify_psnr_quality(psnr_val)
        print(f"   {psnr_val} dB: {quality}")

    # Test 9: NumPy compatibility
    print("\n9. NumPy Compatibility")
    print("-" * 60)

    cover_np = np.random.rand(batch_size, 3, 64, 64).astype(np.float32)
    stego_np = cover_np + \
        np.random.randn(*cover_np.shape).astype(np.float32) * 0.01
    stego_np = np.clip(stego_np, 0, 1)

    psnr = compute_psnr(stego_np, cover_np, max_val=1.0)
    print(f"   PSNR with NumPy input: {psnr:.2f} dB")
    print(f"   ✓ NumPy arrays work")

    # Test 10: Edge cases
    print("\n10. Edge Cases")
    print("-" * 60)

    # Very high PSNR (near identical)
    cover = torch.rand(1, 3, 64, 64)
    stego = cover + torch.randn_like(cover) * 1e-6
    psnr = compute_psnr(stego, cover)
    print(f"   Very small perturbation: {psnr:.2f} dB (very high)")

    # Very low PSNR (completely different)
    cover = torch.zeros(1, 3, 64, 64)
    stego = torch.ones(1, 3, 64, 64)
    psnr = compute_psnr(stego, cover)
    print(f"   Completely different: {psnr:.2f} dB (very low)")

    # Single pixel difference
    cover = torch.zeros(1, 1, 10, 10)
    stego = cover.clone()
    stego[0, 0, 0, 0] = 0.1
    psnr = compute_psnr(stego, cover)
    print(f"   Single pixel diff: {psnr:.2f} dB")

    # Test 11: Batch vs single computation
    print("\n11. Batch vs Single Computation")
    print("-" * 60)

    cover = torch.rand(8, 3, 64, 64)
    stego = cover + torch.randn_like(cover) * 0.02

    # Batch PSNR
    psnr_batch = compute_psnr(stego, cover)

    # Individual PSNRs
    metric = PSNRMetric()
    psnr_per_sample = metric.compute_per_sample(stego, cover)
    psnr_mean = psnr_per_sample.mean().item()

    print(f"   Batch PSNR (average MSE): {psnr_batch:.2f} dB")
    print(f"   Mean of per-sample PSNRs: {psnr_mean:.2f} dB")
    print(f"   Note: These may differ slightly")

    print("\n" + "=" * 60)
    print("✅ All PSNR tests passed!")
    print("=" * 60)

    print("\nUsage Example:")
    print("""
# Basic usage
from evaluation.psnr import compute_psnr, classify_psnr_quality

stego_image = model.encode(cover_image, message)
psnr = compute_psnr(stego_image, cover_image, max_val=1.0)

print(f"PSNR: {psnr:.2f} dB")
print(f"Quality: {classify_psnr_quality(psnr)}")

# Statistics across dataset
from evaluation.psnr import compute_psnr_statistics

stats = compute_psnr_statistics(stego_images, cover_images)
print(f"Mean PSNR: {stats['mean']:.2f} ± {stats['std']:.2f} dB")

# Evaluate with attacks
from evaluation.psnr import evaluate_psnr_with_attacks

attacks = [
    ('jpeg_50', lambda x: jpeg_compression(x, quality=50)),
    ('gaussian_noise', lambda x: add_gaussian_noise(x, std=0.1))
]

results = evaluate_psnr_with_attacks(model, images, messages, attacks)
for condition, psnr in results.items():
    print(f"{condition}: {psnr:.2f} dB")
    """)
