"""Augmentation configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AugmentationConfig:
    """Configuration for image augmentation.

    Attributes
    ----------
    horizontal_flip : bool
        Random horizontal flip.
    vertical_flip : bool
        Random vertical flip.
    rotation_range : float
        Maximum rotation angle in degrees.
    scale_range : tuple[float, float]
        Min/max scale factors.
    translation_range : tuple[float, float]
        Max translation as fraction of image size.
    brightness_range : tuple[float, float]
        Min/max brightness adjustment.
    contrast_range : tuple[float, float]
        Min/max contrast adjustment.
    noise_std : float
        Standard deviation of Gaussian noise.
    blur_range : tuple[float, float]
        Min/max Gaussian blur sigma.
    """

    horizontal_flip: bool = True
    vertical_flip: bool = False
    rotation_range: float = 15.0
    scale_range: tuple[float, float] = (0.9, 1.1)
    translation_range: tuple[float, float] = (0.1, 0.1)
    brightness_range: tuple[float, float] = (0.8, 1.2)
    contrast_range: tuple[float, float] = (0.8, 1.2)
    noise_std: float = 0.0
    blur_range: tuple[float, float] = (0.0, 0.0)
