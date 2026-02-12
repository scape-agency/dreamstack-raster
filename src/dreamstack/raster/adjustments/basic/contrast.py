"""Contrast adjustment function."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dreamstack.raster.core.image import Image


def contrast(image: Image, amount: float = 0) -> Image:
    """
    Adjust image contrast.

    Args:
        image: Input image
        amount: Contrast adjustment (-100 to 100)

    Returns:
        Adjusted image
    """
    from dreamstack.raster.core.pixel import PixelData

    data = image.data.astype(np.float32)
    max_val = 255 if image.bit_depth.name == "UINT8" else 65535
    mid_val = max_val / 2

    # Convert amount to factor
    # At 0, factor is 1
    # At 100, factor approaches infinity
    # At -100, factor approaches 0
    if amount >= 0:
        factor = 1 + amount / 100 * 2
    else:
        factor = 1 + amount / 100

    result = (data - mid_val) * factor + mid_val
    result = np.clip(result, 0, max_val)

    result_image = image.copy()
    result_image._pixel_data = PixelData(
        data=result.astype(image.data.dtype),
        pixel_format=image.pixel_format,
        bit_depth=image.bit_depth,
    )

    return result_image
