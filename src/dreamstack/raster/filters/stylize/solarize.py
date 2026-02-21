# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Solarize
============================

Solarization effect implementation.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dreamstack.raster.core.image import Image


def solarize(image: Image, threshold: int = 128) -> Image:
    """
    Apply solarization effect.

    Args:
        image: Input image
        threshold: Inversion threshold

    Returns:
        Solarized image
    """
    from dreamstack.raster.core.pixel import PixelData

    data = image.data.astype(np.float32)
    max_val = 255 if image.bit_depth.name == "UINT8" else 65535

    # Scale threshold
    thresh = threshold * max_val / 255

    # Invert values above threshold
    result = np.where(data > thresh, max_val - data, data)

    result_image = image.copy()
    result_image._pixel_data = PixelData(  # pylint: disable=protected-access
        data=result.astype(image.data.dtype),
        pixel_format=image.pixel_format,
        bit_depth=image.bit_depth,
    )

    return result_image
