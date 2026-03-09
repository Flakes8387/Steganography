"""
Evaluation metrics for steganography.

Includes:
- BER: Bit Error Rate for message recovery
- PSNR: Peak Signal-to-Noise Ratio for imperceptibility
- SSIM: Structural Similarity Index for perceptual quality
- Steganalysis: CNN-based detection for security evaluation
"""

from .ber import (
    BERMetric,
    compute_ber,
    compute_ber_statistics,
    evaluate_ber_with_attacks
)

from .psnr import (
    PSNRMetric,
    compute_psnr,
    compute_psnr_statistics,
    evaluate_psnr_with_attacks,
    psnr_to_mse,
    mse_to_psnr,
    classify_psnr_quality
)

from .ssim import (
    SSIMMetric,
    MultiScaleSSIM,
    compute_ssim,
    compute_ssim_statistics,
    evaluate_ssim_with_attacks,
    classify_ssim_quality,
    ssim_to_dssim
)

from .steganalysis import (
    SRNet,
    SimpleStegDetector,
    SteganalysisEvaluator,
    evaluate_steganography_security
)

__all__ = [
    # BER
    'BERMetric',
    'compute_ber',
    'compute_ber_statistics',
    'evaluate_ber_with_attacks',

    # PSNR
    'PSNRMetric',
    'compute_psnr',
    'compute_psnr_statistics',
    'evaluate_psnr_with_attacks',
    'psnr_to_mse',
    'mse_to_psnr',
    'classify_psnr_quality',

    # SSIM
    'SSIMMetric',
    'MultiScaleSSIM',
    'compute_ssim',
    'compute_ssim_statistics',
    'evaluate_ssim_with_attacks',
    'classify_ssim_quality',
    'ssim_to_dssim',

    # Steganalysis
    'SRNet',
    'SimpleStegDetector',
    'SteganalysisEvaluator',
    'evaluate_steganography_security',
]
