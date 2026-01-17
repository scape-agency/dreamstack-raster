# -*- coding: utf-8 -*-

"""Desaturate function."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dreamstack.raster.core.image import Image


def desaturate(image: Image, method: str = "luminosity") -> Image:
    """
    Desaturate image to grayscale.

    Args:
        image: Input image
        method: Desaturation method:
            - 'luminosity': Perceptual luminance (default)
            - 'average': Simple average
            - 'lightness': (max + min) / 2
            - 'maximum': Maximum of RGB
            - 'minimum': Minimum of RGB

    Returns:
        Desaturated image
    """
    from dreamstack.raster.core.pixel import PixelData

    if image.channels < 3:
        return image.copy()

    data = image.data.astype(np.float32)
    max_val = 255 if image.bit_depth.name == "UINT8" else 65535

    r = data[:, :, 0]
    g = data[:, :, 1]
    b = data[:, :, 2]

    if method == "luminosity":
        gray = 0.299 * r + 0.587 * g + 0.114 * b
    elif method == "average":
        gray = (r + g + b) / 3
    elif method == "lightness":
        gray = (
            np.maximum(np.maximum(r, g), b) + np.minimum(np.minimum(r, g), b)
        ) / 2
    elif method == "maximum":
        gray = np.maximum(np.maximum(r, g), b)
    elif method == "minimum":
        gray = np.minimum(np.minimum(r, g), b)
    else:
        gray = 0.299 * r + 0.587 * g + 0.114 * b

    gray = np.clip(gray, 0, max_val)

    result = data.copy()
    result[:, :, 0] = gray
    result[:, :, 1] = gray
    result[:, :, 2] = gray

    result_image = image.copy()
    result_image._pixel_data = PixelData(
        data=result.astype(image.data.dtype),
        pixel_format=image.pixel_format,
        bit_depth=image.bit_depth,
    )

    return result_image
