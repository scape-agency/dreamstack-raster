# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - High Pass
=============================

High-pass filter implementation.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

# =============================================================================
# Type Checking Imports
# =============================================================================

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.image import Image


def high_pass(image: Image, radius: float = 10.0) -> Image:
    """
    Apply high-pass filter (useful for overlay sharpening).

    Args:
        image: Input image
        radius: Filter radius

    Returns:
        High-pass filtered image
    """
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.pixel import PixelData

    data = image.data.astype(np.float32)
    max_val = 255 if image.bit_depth.name == "UINT8" else 65535
    mid_val = max_val / 2

    # Gaussian blur
    ksize = int(radius * 2) | 1

    if data.ndim == 3:
        blurred = np.zeros_like(data)
        for i in range(data.shape[2]):
            blurred[:, :, i] = cv2.GaussianBlur(
                data[:, :, i], (ksize, ksize), radius / 3
            )
    else:
        blurred = cv2.GaussianBlur(data, (ksize, ksize), radius / 3)

    # High-pass = original - low-pass + mid gray
    result = data - blurred + mid_val
    result = np.clip(result, 0, max_val)

    result_image = image.copy()
    result_image._pixel_data = PixelData(  # pylint: disable=protected-access
        data=result.astype(image.data.dtype),
        pixel_format=image.pixel_format,
        bit_depth=image.bit_depth,
    )

    return result_image
