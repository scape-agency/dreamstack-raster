# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""HDR toning function."""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

# =============================================================================
# Type Checking Imports
# =============================================================================

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.image import Image


def hdr_toning(
    image: Image,
    strength: float = 50,
    radius: int = 70,
    detail: float = 100,
    shadow: float = 0,
    highlight: float = 0,
    vibrance: float = 20,
    saturation: float = 0,
) -> Image:
    """
    Apply HDR-style toning effect.

    Args:
        image: Input image
        strength: Effect strength (0-200)
        radius: Local adaptation radius
        detail: Detail enhancement (0-300)
        shadow: Shadow adjustment (-100 to 100)
        highlight: Highlight adjustment (-100 to 100)
        vibrance: Vibrance boost (-100 to 100)
        saturation: Saturation adjustment (-100 to 100)

    Returns:
        HDR-toned image
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

    # Local tone mapping
    ksize = radius * 2 + 1
    local_mean = cv2.GaussianBlur(luminance, (ksize, ksize), 0)

    # Detail layer
    detail_layer = luminance - local_mean

    # Tone map
    strength_factor = strength / 100

    # Compress dynamic range
    mapped_lum = np.log1p(luminance * 10) / np.log1p(10)

    # Blend based on strength
    adjusted_lum = (
        luminance * (1 - strength_factor) + mapped_lum * strength_factor
    )

    # Add detail
    detail_factor = detail / 100
    adjusted_lum = adjusted_lum + detail_layer * detail_factor

    # Shadow/highlight
    if shadow != 0 or highlight != 0:
        shadow_mask = np.clip(1 - adjusted_lum / 0.3, 0, 1) ** 2
        highlight_mask = np.clip((adjusted_lum - 0.7) / 0.3, 0, 1) ** 2

        if shadow > 0:
            adjusted_lum = adjusted_lum + shadow_mask * shadow / 100 * (
                1 - adjusted_lum
            )
        else:
            adjusted_lum = adjusted_lum * (1 + shadow / 100 * shadow_mask)

        if highlight > 0:
            adjusted_lum = adjusted_lum + highlight_mask * highlight / 100 * (
                1 - adjusted_lum
            )
        else:
            adjusted_lum = (
                adjusted_lum + highlight_mask * highlight / 100 * adjusted_lum
            )

    adjusted_lum = np.clip(adjusted_lum, 0, 1)

    # Apply to color
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(luminance > 0, adjusted_lum / luminance, 1)

    result = normalized * ratio[:, :, np.newaxis]

    # Vibrance and saturation
    if vibrance != 0 or saturation != 0:
        # Convert to HSV-like
        max_c = np.max(result, axis=2)
        min_c = np.min(result, axis=2)
        current_sat = np.where(max_c > 0, (max_c - min_c) / max_c, 0)

        mean_color = np.mean(result, axis=2, keepdims=True)

        # Vibrance (saturation weighted by inverse saturation)
        if vibrance != 0:
            vib_factor = (
                1 + (vibrance / 100) * (1 - current_sat)[:, :, np.newaxis]
            )
            result = mean_color + (result - mean_color) * vib_factor

        # Saturation
        if saturation != 0:
            sat_factor = 1 + saturation / 100
            result = mean_color + (result - mean_color) * sat_factor

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
