# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Curves adjustment function."""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dreamstack.raster.core.image import Image

from dreamstack.raster.adjustments.curves.curve import Curve


def curves(
    image: Image,
    rgb_curve: Curve | None = None,
    red_curve: Curve | None = None,
    green_curve: Curve | None = None,
    blue_curve: Curve | None = None,
) -> Image:
    """
    Apply curves adjustment.

    Args:
        image: Input image
        rgb_curve: Master RGB curve
        red_curve: Red channel curve
        green_curve: Green channel curve
        blue_curve: Blue channel curve

    Returns:
        Adjusted image
    """
    from dreamstack.raster.core.pixel import PixelData

    data = image.data.astype(np.float32)
    max_val = 255 if image.bit_depth.name == "UINT8" else 65535
    lut_size = 256 if image.bit_depth.name == "UINT8" else 65536

    result = data.copy()

    # Generate lookup tables
    if rgb_curve:
        rgb_lut = rgb_curve.get_lookup_table(lut_size)
    else:
        rgb_lut = None

    channel_curves = [red_curve, green_curve, blue_curve]
    channel_luts = []
    for curve in channel_curves:
        if curve:
            channel_luts.append(curve.get_lookup_table(lut_size))
        else:
            channel_luts.append(None)

    # Apply curves
    if data.ndim == 3:
        for i in range(min(3, data.shape[2])):
            channel = data[:, :, i]

            # Normalize to LUT indices
            indices = (channel / max_val * (lut_size - 1)).astype(int)
            indices = np.clip(indices, 0, lut_size - 1)

            # Apply RGB curve first
            if rgb_lut is not None:
                channel = rgb_lut[indices]
                indices = channel.astype(int)
                indices = np.clip(indices, 0, lut_size - 1)

            # Apply channel curve
            if channel_luts[i] is not None:
                channel = channel_luts[i][indices]
            elif rgb_lut is not None:
                # Channel already processed
                pass
            else:
                channel = data[:, :, i]

            result[:, :, i] = channel * max_val / (lut_size - 1)
    else:
        indices = (data / max_val * (lut_size - 1)).astype(int)
        indices = np.clip(indices, 0, lut_size - 1)

        if rgb_lut is not None:
            result = rgb_lut[indices] * max_val / (lut_size - 1)

    result = np.clip(result, 0, max_val)

    result_image = image.copy()
    result_image._pixel_data = PixelData(  # pylint: disable=protected-access
        data=result.astype(image.data.dtype),
        pixel_format=image.pixel_format,
        bit_depth=image.bit_depth,
    )

    return result_image
