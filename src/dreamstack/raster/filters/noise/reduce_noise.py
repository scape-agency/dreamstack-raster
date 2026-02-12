"""
Dreamstack Raster - Reduce Noise
================================

Noise reduction filter implementation.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from dreamstack.raster.core.image import Image


def reduce_noise(
    image: Image, strength: float = 10, preserve_details: float = 0.5
) -> Image:
    """
    Reduce noise while preserving details.

    Args:
        image: Input image
        strength: Noise reduction strength
        preserve_details: Detail preservation (0-1)

    Returns:
        Denoised image
    """
    from dreamstack.raster.core.pixel import PixelData

    # Use bilateral filter for edge-preserving smoothing
    d = int(strength) | 1
    sigma_color = strength * 10 * (1 - preserve_details)
    sigma_space = strength * 10

    data = image.data

    # Convert to 8-bit for bilateral filter
    if image.bit_depth.name != "UINT8":
        data = (data / data.max() * 255).astype(np.uint8)

    if data.ndim == 3 and data.shape[2] >= 3:
        result = cv2.bilateralFilter(data[:, :, :3], d, sigma_color, sigma_space)
        if data.shape[2] == 4:
            result = np.dstack([result, data[:, :, 3]])
    else:
        if data.ndim == 3:
            data = data[:, :, 0]
        result = cv2.bilateralFilter(data, d, sigma_color, sigma_space)
        result = result[:, :, np.newaxis]

    # Convert back
    if image.bit_depth.name != "UINT8":
        max_val = 65535 if image.bit_depth.name == "UINT16" else 1.0
        result = (result / 255.0 * max_val).astype(image.data.dtype)

    result_image = image.copy()
    result_image._pixel_data = PixelData(
        data=result, pixel_format=image.pixel_format, bit_depth=image.bit_depth
    )

    return result_image
