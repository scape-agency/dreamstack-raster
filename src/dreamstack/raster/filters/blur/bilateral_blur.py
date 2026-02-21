# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Bilateral Blur
==================================

Bilateral blur (edge-preserving smoothing) filter implementation.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from dreamstack.raster.core.image import Image


def bilateral_blur(
    image: Image, d: int = 9, sigma_color: float = 75, sigma_space: float = 75
) -> Image:
    """
    Apply bilateral filter (edge-preserving smoothing).

    Args:
        image: Input image
        d: Diameter of pixel neighborhood
        sigma_color: Filter sigma in the color space
        sigma_space: Filter sigma in the coordinate space

    Returns:
        Blurred image
    """
    from dreamstack.raster.core.pixel import BitDepth, PixelData

    data = image.data

    # Bilateral filter works best on 8-bit
    if image.bit_depth != BitDepth.UINT8:
        data = (data / data.max() * 255).astype(np.uint8)

    if data.ndim == 3 and data.shape[2] >= 3:
        # Use color bilateral
        result = cv2.bilateralFilter(
            data[:, :, :3], d, sigma_color, sigma_space
        )
        if data.shape[2] == 4:
            # Preserve alpha
            result = np.dstack([result, data[:, :, 3]])
    else:
        if data.ndim == 3:
            data = data[:, :, 0]
        result = cv2.bilateralFilter(data, d, sigma_color, sigma_space)
        result = result[:, :, np.newaxis]

    # Convert back to original bit depth
    if image.bit_depth != BitDepth.UINT8:
        max_val = 65535 if image.bit_depth == BitDepth.UINT16 else 1.0
        result = (result / 255.0 * max_val).astype(image.data.dtype)

    result_image = image.copy()
    result_image._pixel_data = PixelData(  # pylint: disable=protected-access
        data=result, pixel_format=image.pixel_format, bit_depth=image.bit_depth
    )

    return result_image
