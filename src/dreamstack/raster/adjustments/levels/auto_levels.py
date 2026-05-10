# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Auto levels adjustment function."""


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


def auto_levels(
    image: Image, clip_black: float = 0.1, clip_white: float = 0.1
) -> Image:
    """
    Auto-adjust levels based on histogram.

    Args:
        image: Input image
        clip_black: Percentage to clip from black
        clip_white: Percentage to clip from white

    Returns:
        Adjusted image
    """
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.pixel import PixelData

    data = image.data.astype(np.float32)
    max_val = image.bit_depth.max_value

    result = data.copy()

    if data.ndim == 3:
        # Process each channel independently
        for i in range(min(3, data.shape[2])):
            channel = data[:, :, i].flatten()

            # Find clip points
            total = len(channel)
            black_clip = int(total * clip_black / 100)
            white_clip = int(total * clip_white / 100)

            sorted_vals = np.sort(channel)
            in_black = sorted_vals[black_clip] if black_clip < total else 0
            in_white = (
                sorted_vals[-white_clip - 1] if white_clip < total else max_val
            )

            if in_white > in_black:
                result[:, :, i] = (
                    (data[:, :, i] - in_black)
                    / (in_white - in_black)
                    * max_val
                )
    else:
        flat = data.flatten()
        total = len(flat)
        black_clip = int(total * clip_black / 100)
        white_clip = int(total * clip_white / 100)

        sorted_vals = np.sort(flat)
        in_black = sorted_vals[black_clip] if black_clip < total else 0
        in_white = (
            sorted_vals[-white_clip - 1] if white_clip < total else max_val
        )

        if in_white > in_black:
            result = (data - in_black) / (in_white - in_black) * max_val

    result = np.clip(result, 0, max_val)

    result_image = image.copy()
    result_image._pixel_data = PixelData(  # pylint: disable=protected-access
        data=result.astype(image.data.dtype),
        pixel_format=image.pixel_format,
        bit_depth=image.bit_depth,
    )

    return result_image
