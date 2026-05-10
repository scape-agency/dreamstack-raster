# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""Split toning function."""


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


def split_toning(
    image: Image,
    shadow_hue: float = 0,
    shadow_saturation: float = 0,
    highlight_hue: float = 0,
    highlight_saturation: float = 0,
    balance: float = 0,
) -> Image:
    """
    Apply split toning effect.

    Args:
        image: Input image
        shadow_hue: Shadow color hue (0-360)
        shadow_saturation: Shadow color saturation (0-100)
        highlight_hue: Highlight color hue (0-360)
        highlight_saturation: Highlight color saturation (0-100)
        balance: Shadow/highlight balance (-100 to 100)

    Returns:
        Split-toned image
    """
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.pixel import PixelData

    if image.channels < 3:
        return image.copy()

    data = image.data.astype(np.float32)
    max_val = image.bit_depth.max_value

    normalized = data[:, :, :3] / max_val

    # Calculate luminance
    luminance = (
        0.299 * normalized[:, :, 0]
        + 0.587 * normalized[:, :, 1]
        + 0.114 * normalized[:, :, 2]
    )

    # Create shadow and highlight masks
    balance_shift = balance / 200  # -0.5 to 0.5
    mid_point = 0.5 + balance_shift

    shadow_mask = np.clip(1 - luminance / mid_point, 0, 1)
    highlight_mask = np.clip((luminance - mid_point) / (1 - mid_point), 0, 1)

    # Convert hues to RGB
    def hue_to_rgb(hue):
        h = hue / 60
        x = 1 - abs(h % 2 - 1)

        if h < 1:
            return np.array([1, x, 0])
        elif h < 2:
            return np.array([x, 1, 0])
        elif h < 3:
            return np.array([0, 1, x])
        elif h < 4:
            return np.array([0, x, 1])
        elif h < 5:
            return np.array([x, 0, 1])
        else:
            return np.array([1, 0, x])

    shadow_color = hue_to_rgb(shadow_hue % 360)
    highlight_color = hue_to_rgb(highlight_hue % 360)

    # Apply toning
    result = normalized.copy()

    # Shadow toning
    if shadow_saturation > 0:
        shadow_intensity = shadow_mask * shadow_saturation / 100
        for i in range(3):
            # Blend towards shadow color
            result[:, :, i] = result[:, :, i] + shadow_intensity * (
                shadow_color[i] - 0.5
            ) * (1 - luminance)

    # Highlight toning
    if highlight_saturation > 0:
        highlight_intensity = highlight_mask * highlight_saturation / 100
        for i in range(3):
            # Blend towards highlight color
            result[:, :, i] = (
                result[:, :, i]
                + highlight_intensity * (highlight_color[i] - 0.5) * luminance
            )

    result = np.clip(result, 0, 1)

    final = data.copy()
    final[:, :, :3] = result * max_val

    result_image = image.copy()
    result_image._pixel_data = PixelData(  # pylint: disable=protected-access
        data=final.astype(image.data.dtype),
        pixel_format=image.pixel_format,
        bit_depth=image.bit_depth,
    )

    return result_image
