# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Sketch
==========================

Pencil sketch effect implementation.

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


def sketch(image: Image, detail: float = 0.5, stroke_width: int = 1) -> Image:
    """
    Apply pencil sketch effect.

    Args:
        image: Input image
        detail: Detail level (0-1)
        stroke_width: Width of sketch strokes

    Returns:
        Sketch image
    """
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.pixel import PixelData

    data = image.data

    if image.bit_depth.name != "UINT8":
        data = (data / data.max() * 255).astype(np.uint8)

    # Convert to grayscale
    if data.ndim == 3 and data.shape[2] >= 3:
        gray = cv2.cvtColor(data[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:
        gray = data[:, :, 0] if data.ndim == 3 else data

    # Invert
    inverted = 255 - gray

    # Blur the inverted image
    blur_size = int((1 - detail) * 50) | 1
    blurred = cv2.GaussianBlur(inverted, (blur_size, blur_size), 0)

    # Color dodge blend
    result = cv2.divide(gray, 255 - blurred, scale=256)

    # Add stroke width effect
    if stroke_width > 1:
        kernel = np.ones((stroke_width, stroke_width), np.uint8)
        result = cv2.erode(result, kernel)

    result = result[:, :, np.newaxis]

    if image.bit_depth.name != "UINT8":
        max_val = 65535 if image.bit_depth.name == "UINT16" else 1.0
        result = (result / 255.0 * max_val).astype(image.data.dtype)

    result_image = image.copy()
    result_image._pixel_data = PixelData(  # pylint: disable=protected-access
        data=result,
        pixel_format=image.to_grayscale().pixel_format,
        bit_depth=image.bit_depth,
    )

    return result_image
