# -*- coding: utf-8 -*-

"""
Dreamstack Raster - Posterize
=============================

Posterize effect implementation.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dreamstack.raster.core.image import Image


def posterize(image: Image, levels: int = 4) -> Image:
    """
    Reduce color levels (posterize effect).

    Args:
        image: Input image
        levels: Number of levels per channel (2-255)

    Returns:
        Posterized image
    """
    from dreamstack.raster.core.pixel import PixelData

    data = image.data.astype(np.float32)
    max_val = 255 if image.bit_depth.name == "UINT8" else 65535

    # Quantize
    levels = max(2, min(255, levels))
    step = max_val / levels

    result = np.floor(data / step) * step
    result = np.clip(result, 0, max_val)

    result_image = image.copy()
    result_image._pixel_data = PixelData(
        data=result.astype(image.data.dtype),
        pixel_format=image.pixel_format,
        bit_depth=image.bit_depth,
    )

    return result_image
