"""
Dreamstack Raster - Pixelate
============================

Pixelation effect implementation.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    from dreamstack.raster.core.image import Image


def pixelate(image: Image, block_size: int = 10) -> Image:
    """
    Apply pixelation effect.

    Args:
        image: Input image
        block_size: Size of pixel blocks

    Returns:
        Pixelated image
    """
    from dreamstack.raster.core.pixel import PixelData

    data = image.data
    h, w = data.shape[:2]

    # Downscale
    small_h = max(1, h // block_size)
    small_w = max(1, w // block_size)

    small = cv2.resize(data, (small_w, small_h), interpolation=cv2.INTER_AREA)

    # Upscale with nearest neighbor
    result = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    result_image = image.copy()
    result_image._pixel_data = PixelData(
        data=result, pixel_format=image.pixel_format, bit_depth=image.bit_depth
    )

    return result_image
