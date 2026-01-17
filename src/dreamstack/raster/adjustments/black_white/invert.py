# -*- coding: utf-8 -*-

"""Invert colors function."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dreamstack.raster.core.image import Image


def invert(image: Image) -> Image:
    """
    Invert image colors.

    Args:
        image: Input image

    Returns:
        Inverted image
    """
    from dreamstack.raster.core.pixel import PixelData

    data = image.data
    max_val = 255 if image.bit_depth.name == "UINT8" else 65535

    result = max_val - data

    # Don't invert alpha channel
    if data.ndim == 3 and data.shape[2] == 4:
        result[:, :, 3] = data[:, :, 3]

    result_image = image.copy()
    result_image._pixel_data = PixelData(
        data=result, pixel_format=image.pixel_format, bit_depth=image.bit_depth
    )

    return result_image
