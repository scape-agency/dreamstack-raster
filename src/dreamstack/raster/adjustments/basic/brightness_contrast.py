# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Brightness and contrast combined adjustment function."""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.image import Image


def brightness_contrast(
    image: Image,
    brightness_amount: float = 0,
    contrast_amount: float = 0,
    legacy: bool = False,
) -> Image:
    """
    Adjust brightness and contrast together.

    Args:
        image: Input image
        brightness_amount: Brightness (-100 to 100)
        contrast_amount: Contrast (-100 to 100)
        legacy: Use legacy algorithm

    Returns:
        Adjusted image
    """
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.pixel import PixelData

    data = image.data.astype(np.float32)
    max_val = 255 if image.bit_depth.name == "UINT8" else 65535

    if legacy:
        # Legacy Photoshop algorithm
        if brightness_amount > 0:
            shadow = brightness_amount
            highlight = max_val
        else:
            shadow = 0
            highlight = max_val + brightness_amount

        if highlight - shadow > 0:
            result = (data - shadow) * max_val / (highlight - shadow)
        else:
            result = data

        if contrast_amount > 0:
            contrast_factor = 1 + contrast_amount / 100
            result = (result - max_val / 2) * contrast_factor + max_val / 2
        elif contrast_amount < 0:
            contrast_factor = 1 + contrast_amount / 100
            result = (result - max_val / 2) * contrast_factor + max_val / 2
    else:
        # Modern algorithm using curves
        # Brightness shifts midpoint
        brightness_offset = brightness_amount * max_val / 100

        # Contrast adjusts slope around midpoint
        if contrast_amount >= 0:
            contrast_factor = 1 + contrast_amount / 100 * 2
        else:
            contrast_factor = 1 + contrast_amount / 100

        mid_val = max_val / 2
        result = (
            (data - mid_val) * contrast_factor + mid_val + brightness_offset
        )

    result = np.clip(result, 0, max_val)

    result_image = image.copy()

    result_image._pixel_data = PixelData(  # pylint: disable=protected-access
        data=result.astype(image.data.dtype),
        pixel_format=image.pixel_format,
        bit_depth=image.bit_depth,
    )

    return result_image
