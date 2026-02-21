# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Prewitt Edge Detection
==========================================

Prewitt edge detection implementation.

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


def prewitt(image: Image) -> Image:
    """
    Apply Prewitt edge detection.

    Args:
        image: Input image

    Returns:
        Edge image
    """
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.pixel import PixelData

    gray_img = image.to_grayscale() if image.channels > 1 else image
    data = gray_img.data.astype(np.float32)

    if data.ndim == 3:
        data = data[:, :, 0]

    # Prewitt kernels
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)

    gx = cv2.filter2D(data, -1, kernel_x)
    gy = cv2.filter2D(data, -1, kernel_y)

    magnitude = np.sqrt(gx**2 + gy**2)

    max_val = 255 if image.bit_depth.name == "UINT8" else 65535
    magnitude = np.clip(magnitude, 0, max_val)

    result_image = image.copy()
    result_image._pixel_data = PixelData(  # pylint: disable=protected-access
        data=magnitude[:, :, np.newaxis].astype(image.data.dtype),
        pixel_format=gray_img.pixel_format,
        bit_depth=image.bit_depth,
    )

    return result_image
