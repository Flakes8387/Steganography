"""
Defense mechanisms for robust steganography.

Includes anti-aliasing, denoising, and adversarial training modules
to improve robustness against attacks and distortions.
"""

# Anti-aliasing filters
from .antialias import (
    GaussianAntiAlias,
    BilateralAntiAlias,
    MedianAntiAlias,
    AdaptiveAntiAlias
)

# Denoising filters
from .denoise import (
    GaussianDenoise,
    NonLocalMeansDenoise,
    WaveletDenoise,
    CNNDenoise,
    DenoiseBeforeDecode
)

# Adversarial training
from .adversarial import (
    FGSM,
    PGD,
    AdversarialTraining,
    AdversarialAugmentation
)

__all__ = [
    # Anti-aliasing
    'GaussianAntiAlias',
    'BilateralAntiAlias',
    'MedianAntiAlias',
    'AdaptiveAntiAlias',

    # Denoising
    'GaussianDenoise',
    'NonLocalMeansDenoise',
    'WaveletDenoise',
    'CNNDenoise',
    'DenoiseBeforeDecode',

    # Adversarial training
    'FGSM',
    'PGD',
    'AdversarialTraining',
    'AdversarialAugmentation',
]
