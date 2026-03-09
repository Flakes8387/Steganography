"""
Attack/Distortion modules for steganography robustness testing.

These modules implement differentiable augmentations that simulate
real-world distortions (social media compression, image processing, etc.)
"""

from .jpeg import JPEGCompression
from .noise import GaussianNoise, SaltPepperNoise, SpeckleNoise, CombinedNoise
from .resize import ResizeAttack, AdaptiveResize, RandomResize
from .blur import GaussianBlur, MotionBlur, AverageBlur, CombinedBlur
from .crop import RandomCrop, CenterCrop, PadCrop, AspectRatioCrop, CombinedCrop
from .color_jitter import (
    BrightnessJitter,
    ContrastJitter,
    SaturationJitter,
    HueJitter,
    ColorJitter
)

__all__ = [
    # JPEG
    'JPEGCompression',

    # Noise
    'GaussianNoise',
    'SaltPepperNoise',
    'SpeckleNoise',
    'CombinedNoise',

    # Resize
    'ResizeAttack',
    'AdaptiveResize',
    'RandomResize',

    # Blur
    'GaussianBlur',
    'MotionBlur',
    'AverageBlur',
    'CombinedBlur',

    # Crop
    'RandomCrop',
    'CenterCrop',
    'PadCrop',
    'AspectRatioCrop',
    'CombinedCrop',

    # Color Jitter
    'BrightnessJitter',
    'ContrastJitter',
    'SaturationJitter',
    'HueJitter',
    'ColorJitter',
]
