# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Unsharp Mask
================================

Unsharp mask sharpening filter implementation.

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


def unsharp_mask(
    image: Image, amount: float = 100, radius: float = 1.0, threshold: int = 0
) -> Image:
    """
    Apply unsharp mask sharpening.

    Args:
        image: Input image
        amount: Sharpening amount (0-500%)
        radius: Blur radius
        threshold: Threshold for sharpening (reduces noise)

    Returns:
        Sharpened image
    """
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.pixel import PixelData

    data = image.data.astype(np.float32)
    max_val = 255 if image.bit_depth.name == "UINT8" else 65535

    # Gaussian blur
    ksize = int(radius * 2) | 1
    ksize = max(ksize, 3)

    if data.ndim == 3:
        blurred = np.zeros_like(data)
        for i in range(data.shape[2]):
            blurred[:, :, i] = cv2.GaussianBlur(
                data[:, :, i], (ksize, ksize), radius / 3
            )
    else:
        blurred = cv2.GaussianBlur(data, (ksize, ksize), radius / 3)

    # Calculate mask
    mask = data - blurred

    # Apply threshold
    if threshold > 0:
        abs_mask = np.abs(mask)
        mask = np.where(abs_mask > threshold, mask, 0)

    # Apply sharpening
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
