# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Auto contrast adjustment function."""


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


def auto_contrast(image: Image, clip: float = 0.1) -> Image:
    """
    Auto-adjust contrast.

    Args:
        image: Input image
        clip: Percentage to clip

    Returns:
        Adjusted image
    """
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.pixel import PixelData

    data = image.data.astype(np.float32)
    max_val = image.bit_depth.max_value

    # Find global min/max
    if data.ndim == 3:
        luminance = (
            0.299 * data[:, :, 0]
            + 0.587 * data[:, :, 1]
            + 0.114 * data[:, :, 2]
        )
    else:
        luminance = data

    flat = luminance.flatten()
    total = len(flat)
    clip_count = int(total * clip / 100)

    sorted_vals = np.sort(flat)
    in_black = sorted_vals[clip_count] if clip_count < total else 0
    in_white = sorted_vals[-clip_count - 1] if clip_count < total else max_val

    if in_white <= in_black:
        return image.copy()

    # Apply same contrast adjustment to all channels
    scale = max_val / (in_white - in_black)
    result = (data - in_black) * scale
    result = np.clip(result, 0, max_val)

    result_image = image.copy()
    result_image._pixel_data = PixelData(  # pylint: disable=protected-access
        data=result.astype(image.data.dtype),
        pixel_format=image.pixel_format,
        bit_depth=image.bit_depth,
    )

    return result_image
