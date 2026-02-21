# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Levels adjustment function."""


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


def levels(
    image: Image,
    input_black: float = 0,
    input_white: float = 255,
    input_gamma: float = 1.0,
    output_black: float = 0,
    output_white: float = 255,
    channel: str | None = None,
) -> Image:
    """
    Apply levels adjustment.

    Args:
        image: Input image
        input_black: Input black point (0-255)
        input_white: Input white point (0-255)
        input_gamma: Midtone gamma
        output_black: Output black point (0-255)
        output_white: Output white point (0-255)
        channel: Specific channel ('red', 'green', 'blue', 'rgb', None for all)

    Returns:
        Adjusted image
    """
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.pixel import PixelData

    data = image.data.astype(np.float32)
    max_val = 255 if image.bit_depth.name == "UINT8" else 65535

    # Scale parameters to bit depth
    scale = max_val / 255
    in_black = input_black * scale
    in_white = input_white * scale
    out_black = output_black * scale
    out_white = output_white * scale

    def apply_levels(channel_data):
        # Input levels
        result = (channel_data - in_black) / (in_white - in_black)
        result = np.clip(result, 0, 1)

        # Gamma
        result = np.power(result, 1 / input_gamma)

        # Output levels
        result = result * (out_white - out_black) + out_black

        return result

    result = data.copy()

    if channel is None or channel == "rgb":
        # Apply to all channels
        if data.ndim == 3:
            for i in range(min(3, data.shape[2])):
                result[:, :, i] = apply_levels(data[:, :, i])
        else:
            result = apply_levels(data)
    else:
        # Apply to specific channel
        channel_idx = {"red": 0, "green": 1, "blue": 2}.get(channel.lower())
        if (
            channel_idx is not None
            and data.ndim == 3
            and channel_idx < data.shape[2]
        ):
            result[:, :, channel_idx] = apply_levels(data[:, :, channel_idx])

    result = np.clip(result, 0, max_val)

    result_image = image.copy()
    result_image._pixel_data = PixelData(  # pylint: disable=protected-access
        data=result.astype(image.data.dtype),
        pixel_format=image.pixel_format,
        bit_depth=image.bit_depth,
    )

    return result_image
