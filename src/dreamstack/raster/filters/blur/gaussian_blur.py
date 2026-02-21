# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Gaussian Blur
=================================

Gaussian blur filter implementation.

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
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.image import Image


def gaussian_blur(
    image: Image, radius: float = 5.0, sigma: float | None = None
) -> Image:
    """
    Apply Gaussian blur.

    Args:
        image: Input image
        radius: Blur radius in pixels
        sigma: Standard deviation (auto-calculated if None)

    Returns:
        Blurred image
    """
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.pixel import PixelData

    if sigma is None:
        sigma = radius / 3.0

    # Kernel size must be odd
    ksize = int(radius * 2) | 1

    data = image.data.astype(np.float32)

    # Process each channel
    if data.ndim == 3:
        result = np.zeros_like(data)
        for i in range(data.shape[2]):
            result[:, :, i] = cv2.GaussianBlur(
                data[:, :, i], (ksize, ksize), sigma
            )
    else:
        result = cv2.GaussianBlur(data, (ksize, ksize), sigma)

    result_image = image.copy()
    result_image._pixel_data = PixelData(  # pylint: disable=protected-access
        data=result.astype(image.data.dtype),
        pixel_format=image.pixel_format,
        bit_depth=image.bit_depth,
    )

    return result_image
