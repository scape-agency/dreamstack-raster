# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Box Blur
============================

Box (mean) blur filter implementation.

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


def box_blur(image: Image, radius: int = 5) -> Image:
    """
    Apply box (mean) blur.

    Args:
        image: Input image
        radius: Blur radius in pixels

    Returns:
        Blurred image
    """
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.pixel import PixelData

    ksize = radius * 2 + 1

    data = image.data.astype(np.float32)

    if data.ndim == 3:
        result = np.zeros_like(data)
        for i in range(data.shape[2]):
            result[:, :, i] = cv2.blur(data[:, :, i], (ksize, ksize))
    else:
        result = cv2.blur(data, (ksize, ksize))

    result_image = image.copy()
    result_image._pixel_data = PixelData(  # pylint: disable=protected-access
        data=result.astype(image.data.dtype),
        pixel_format=image.pixel_format,
        bit_depth=image.bit_depth,
    )

    return result_image
