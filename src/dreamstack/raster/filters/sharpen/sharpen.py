# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Sharpen
===========================

Simple kernel sharpening filter implementation.

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


def sharpen(image: Image, amount: float = 1.0) -> Image:
    """
    Apply simple kernel sharpening.

    Args:
        image: Input image
        amount: Sharpening amount (0.0-2.0)

    Returns:
        Sharpened image
    """
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.pixel import PixelData

    # Sharpening kernel
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)

    # Blend between identity and sharpen kernel
    identity = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)

    kernel = identity * (1 - amount) + kernel * amount

    data = image.data.astype(np.float32)
    max_val = 255 if image.bit_depth.name == "UINT8" else 65535

    if data.ndim == 3:
        result = np.zeros_like(data)
        for i in range(data.shape[2]):
            result[:, :, i] = cv2.filter2D(data[:, :, i], -1, kernel)
    else:
        result = cv2.filter2D(data, -1, kernel)

    result = np.clip(result, 0, max_val)

    result_image = image.copy()
    result_image._pixel_data = PixelData(  # pylint: disable=protected-access
        data=result.astype(image.data.dtype),
        pixel_format=image.pixel_format,
        bit_depth=image.bit_depth,
    )

    return result_image
