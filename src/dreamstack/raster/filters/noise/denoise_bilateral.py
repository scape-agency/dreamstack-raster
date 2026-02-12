"""
Dreamstack Raster - Denoise Bilateral
=====================================

Bilateral filter denoising implementation.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dreamstack.raster.core.image import Image


def denoise_bilateral(
    image: Image, d: int = 9, sigma_color: float = 75, sigma_space: float = 75
) -> Image:
    """
    Denoise using bilateral filter.

    Args:
        image: Input image
        d: Diameter of pixel neighborhood
        sigma_color: Filter sigma in color space
        sigma_space: Filter sigma in coordinate space

    Returns:
        Denoised image
    """
    from dreamstack.raster.filters.blur.bilateral_blur import bilateral_blur

    return bilateral_blur(image, d, sigma_color, sigma_space)
