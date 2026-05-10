# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Auto color adjustment function."""


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

from dreamstack.raster.adjustments.levels.auto_contrast import auto_contrast


def auto_color(
    image: Image,
    clip: float = 0.5,
    neutral: tuple[float, float, float] | None = None,
) -> Image:
    """
    Auto-adjust color balance.

    Args:
        image: Input image
        clip: Percentage to clip
        neutral: Target neutral color (R, G, B) or None for auto

    Returns:
        Adjusted image
    """
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.pixel import PixelData

    if image.channels < 3:
        return auto_contrast(image, clip)

    data = image.data.astype(np.float32)
    max_val = image.bit_depth.max_value

    result = data.copy()

    # Find clip points for each channel
    for i in range(3):
        channel = data[:, :, i].flatten()
        total = len(channel)
        clip_count = int(total * clip / 100)

        sorted_vals = np.sort(channel)
        in_black = sorted_vals[clip_count] if clip_count < total else 0
        in_white = (
            sorted_vals[-clip_count - 1] if clip_count < total else max_val
        )

        if in_white > in_black:
            result[:, :, i] = (
                (data[:, :, i] - in_black) / (in_white - in_black) * max_val
            )

    # Apply neutral balance
    if neutral:
        target = np.array(neutral) * max_val / 255
        current_mean = np.array([result[:, :, i].mean() for i in range(3)])

        for i in range(3):
            if current_mean[i] > 0:
                result[:, :, i] = result[:, :, i] * target[i] / current_mean[i]

    result = np.clip(result, 0, max_val)

    result_image = image.copy()
    result_image._pixel_data = PixelData(  # pylint: disable=protected-access
        data=result.astype(image.data.dtype),
        pixel_format=image.pixel_format,
        bit_depth=image.bit_depth,
    )

    return result_image
