# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""Saturation adjustment function."""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

# =============================================================================
# Type Checking Imports
# =============================================================================

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.image import Image

from dreamstack.raster.adjustments.basic._color_utils import (
    _hsv_to_rgb,
    _rgb_to_hsv,
)


def saturation(image: Image, amount: float = 0) -> Image:
    """
    Adjust color saturation.

    Args:
        image: Input image
        amount: Saturation (-100 to 100)

    Returns:
        Adjusted image
    """
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.pixel import PixelData

    if image.channels < 3:
        return image.copy()

    data = image.data.astype(np.float32)
    max_val = image.bit_depth.max_value

    # Convert to HSV
    hsv = _rgb_to_hsv(data[:, :, :3] / max_val)

    # Adjust saturation
    factor = 1 + amount / 100
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 1)

    # Convert back to RGB
    rgb = _hsv_to_rgb(hsv) * max_val

    result = data.copy()
    result[:, :, :3] = rgb
    result = np.clip(result, 0, max_val)

    result_image = image.copy()
    result_image._pixel_data = PixelData(  # pylint: disable=protected-access
        data=result.astype(image.data.dtype),
        pixel_format=image.pixel_format,
        bit_depth=image.bit_depth,
    )

    return result_image
