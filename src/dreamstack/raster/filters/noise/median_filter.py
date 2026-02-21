# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Median Filter
=================================

Median filter implementation.

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


def median_filter(image: Image, size: int = 3) -> Image:
    """
    Apply median filter (good for salt-and-pepper noise).

    Args:
        image: Input image
        size: Filter kernel size (odd number)

    Returns:
        Filtered image
    """
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.pixel import PixelData

    size = size | 1  # Ensure odd

    data = image.data

    if data.ndim == 3:
        result = np.zeros_like(data)
        for i in range(data.shape[2]):
            result[:, :, i] = cv2.medianBlur(data[:, :, i], size)
    else:
        result = cv2.medianBlur(data, size)

    result_image = image.copy()
    result_image._pixel_data = PixelData(  # pylint: disable=protected-access
        data=result, pixel_format=image.pixel_format, bit_depth=image.bit_depth
    )

    return result_image
