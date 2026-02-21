# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Channel mixer function."""


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


def channel_mixer(
    image: Image,
    output_channel: str = "gray",
    red: float = 100,
    green: float = 0,
    blue: float = 0,
    constant: float = 0,
    monochrome: bool = False,
) -> Image:
    """
    Mix color channels.

    Args:
        image: Input image
        output_channel: Target channel ('red', 'green', 'blue', 'gray')
        red: Red source percentage (-200 to 200)
        green: Green source percentage (-200 to 200)
        blue: Blue source percentage (-200 to 200)
        constant: Constant offset (-200 to 200)
        monochrome: Output monochrome (uses same mix for all channels)

    Returns:
        Channel-mixed image
    """
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.pixel import PixelData

    if image.channels < 3:
        return image.copy()

    data = image.data.astype(np.float32)
    max_val = 255 if image.bit_depth.name == "UINT8" else 65535

    r = data[:, :, 0]
    g = data[:, :, 1]
    b = data[:, :, 2]

    # Calculate mixed channel
    mixed = (
        r * red / 100
        + g * green / 100
        + b * blue / 100
        + constant * max_val / 100
    )
    mixed = np.clip(mixed, 0, max_val)

    result = data.copy()

    if monochrome or output_channel == "gray":
        result[:, :, 0] = mixed
        result[:, :, 1] = mixed
        result[:, :, 2] = mixed
    else:
        channel_idx = {"red": 0, "green": 1, "blue": 2}.get(
            output_channel.lower(), 0
        )
        result[:, :, channel_idx] = mixed

    result_image = image.copy()
    result_image._pixel_data = PixelData(  # pylint: disable=protected-access
        data=result.astype(image.data.dtype),
        pixel_format=image.pixel_format,
        bit_depth=image.bit_depth,
    )

    return result_image
