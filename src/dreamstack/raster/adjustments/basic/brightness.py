"""Brightness adjustment function."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dreamstack.raster.core.image import Image


def brightness(image: Image, amount: float = 0) -> Image:
    """
    Adjust image brightness.

    Args:
        image: Input image
        amount: Brightness adjustment (-100 to 100)

    Returns:
        Adjusted image
    """
    from dreamstack.raster.core.pixel import PixelData

    data = image.data.astype(np.float32)
    max_val = 255 if image.bit_depth.name == "UINT8" else 65535

    # Scale amount
    offset = amount * max_val / 100

    result = data + offset
    result = np.clip(result, 0, max_val)

    result_image = image.copy()
    result_image._pixel_data = PixelData(
        data=result.astype(image.data.dtype),
        pixel_format=image.pixel_format,
        bit_depth=image.bit_depth,
    )

    return result_image
