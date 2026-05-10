# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""Hue/Saturation adjustment function."""


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

from dreamstack.raster.adjustments.color_balance._color_utils import (
    _hsl_to_rgb,
    _rgb_to_hsl,
)


def hue_saturation(
    image: Image,
    hue: float = 0,
    saturation: float = 0,
    lightness: float = 0,
    colorize: bool = False,
    colorize_hue: float = 0,
    colorize_saturation: float = 50,
) -> Image:
    """
    Adjust hue, saturation, and lightness.

    Args:
        image: Input image
        hue: Hue shift (-180 to 180)
        saturation: Saturation adjustment (-100 to 100)
        lightness: Lightness adjustment (-100 to 100)
        colorize: Apply colorize effect
        colorize_hue: Hue for colorize (0-360)
        colorize_saturation: Saturation for colorize (0-100)

    Returns:
        Adjusted image
    """
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.pixel import PixelData

    if image.channels < 3:
        return image.copy()

    data = image.data.astype(np.float32)
    max_val = image.bit_depth.max_value

    # Convert to HSL
    normalized = data[:, :, :3] / max_val
    hsl = _rgb_to_hsl(normalized)

    if colorize:
        # Colorize mode
        hsl[:, :, 0] = colorize_hue / 360.0
        hsl[:, :, 1] = colorize_saturation / 100.0

        # Adjust lightness
        if lightness > 0:
            hsl[:, :, 2] = hsl[:, :, 2] + (1 - hsl[:, :, 2]) * lightness / 100
        elif lightness < 0:
            hsl[:, :, 2] = hsl[:, :, 2] * (1 + lightness / 100)
    else:
        # Normal adjustment
        # Hue shift
        hsl[:, :, 0] = (hsl[:, :, 0] + hue / 360.0) % 1.0

        # Saturation
        if saturation > 0:
            hsl[:, :, 1] = hsl[:, :, 1] + (1 - hsl[:, :, 1]) * saturation / 100
        else:
            hsl[:, :, 1] = hsl[:, :, 1] * (1 + saturation / 100)

        # Lightness
        if lightness > 0:
            hsl[:, :, 2] = hsl[:, :, 2] + (1 - hsl[:, :, 2]) * lightness / 100
        elif lightness < 0:
            hsl[:, :, 2] = hsl[:, :, 2] * (1 + lightness / 100)

    hsl = np.clip(hsl, 0, 1)

    # Convert back to RGB
    rgb = _hsl_to_rgb(hsl)

    result = data.copy()
    result[:, :, :3] = rgb * max_val
    result = np.clip(result, 0, max_val)

    result_image = image.copy()
    result_image._pixel_data = PixelData(  # pylint: disable=protected-access
        data=result.astype(image.data.dtype),
        pixel_format=image.pixel_format,
        bit_depth=image.bit_depth,
    )

    return result_image
