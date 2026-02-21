# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Clarity
===========================

Clarity enhancement (midtone contrast) filter implementation.

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


def clarity(image: Image, amount: float = 50) -> Image:
    """
    Apply clarity enhancement (midtone contrast).

    Args:
        image: Input image
        amount: Clarity amount (-100 to 100)

    Returns:
        Enhanced image
    """
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.pixel import PixelData

    data = image.data.astype(np.float32)
    max_val = 255 if image.bit_depth.name == "UINT8" else 65535

    # High-pass at large radius
    ksize = 41

    if data.ndim == 3:
        blurred = np.zeros_like(data)
        for i in range(data.shape[2]):
            blurred[:, :, i] = cv2.GaussianBlur(
                data[:, :, i], (ksize, ksize), 0
            )
    else:
        blurred = cv2.GaussianBlur(data, (ksize, ksize), 0)

    # High-pass mask
    mask = data - blurred

    # Apply with amount
    amount_factor = amount / 100.0
    result = data + mask * amount_factor

    result = np.clip(result, 0, max_val)

    result_image = image.copy()
    result_image._pixel_data = PixelData(  # pylint: disable=protected-access
        data=result.astype(image.data.dtype),
        pixel_format=image.pixel_format,
        bit_depth=image.bit_depth,
    )

    return result_image
