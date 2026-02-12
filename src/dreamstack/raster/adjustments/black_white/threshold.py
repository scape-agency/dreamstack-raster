"""Threshold function."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dreamstack.raster.core.image import Image


def threshold(image: Image, level: int = 128) -> Image:
    """
    Apply threshold to create black and white image.

    Args:
        image: Input image
        level: Threshold level (0-255)

    Returns:
        Thresholded image
    """
    from dreamstack.raster.core.pixel import PixelData

    data = image.data.astype(np.float32)
    max_val = 255 if image.bit_depth.name == "UINT8" else 65535

    # Scale threshold
    thresh = level * max_val / 255

    if data.ndim == 3 and data.shape[2] >= 3:
        luminance = (
            0.299 * data[:, :, 0] + 0.587 * data[:, :, 1] + 0.114 * data[:, :, 2]
        )
    else:
        luminance = data[:, :, 0] if data.ndim == 3 else data

    binary = np.where(luminance > thresh, max_val, 0)

    result = image.data.copy()
    if result.ndim == 3:
        result[:, :, 0] = binary
        result[:, :, 1] = binary
        result[:, :, 2] = binary
    else:
        result = binary

    result_image = image.copy()
    result_image._pixel_data = PixelData(
        data=result.astype(image.data.dtype),
        pixel_format=image.pixel_format,
        bit_depth=image.bit_depth,
    )

    return result_image
