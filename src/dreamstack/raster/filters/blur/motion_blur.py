# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Motion Blur
===============================

Motion blur filter implementation.

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


def motion_blur(image: Image, size: int = 15, angle: float = 0) -> Image:
    """
    Apply motion blur.

    Args:
        image: Input image
        size: Blur length in pixels
        angle: Blur angle in degrees

    Returns:
        Blurred image
    """
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.pixel import PixelData

    # Create motion blur kernel
    kernel = np.zeros((size, size))
    center = size // 2

    # Calculate line endpoints
    angle_rad = np.radians(angle)
    dx = np.cos(angle_rad)
    dy = np.sin(angle_rad)

    for i in range(size):
        t = (i - center) / center if center > 0 else 0
        x = int(center + t * center * dx)
        y = int(center + t * center * dy)
        if 0 <= x < size and 0 <= y < size:
            kernel[y, x] = 1

    kernel /= kernel.sum()

    data = image.data.astype(np.float32)

    if data.ndim == 3:
        result = np.zeros_like(data)
        for i in range(data.shape[2]):
            result[:, :, i] = cv2.filter2D(data[:, :, i], -1, kernel)
    else:
        result = cv2.filter2D(data, -1, kernel)

    result_image = image.copy()
    result_image._pixel_data = PixelData(  # pylint: disable=protected-access
        data=result.astype(image.data.dtype),
        pixel_format=image.pixel_format,
        bit_depth=image.bit_depth,
    )

    return result_image
