# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""Shadows and highlights adjustment function."""


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


def shadows_highlights(
    image: Image,
    shadows: float = 0,
    highlights: float = 0,
    shadow_tonal_width: float = 50,
    highlight_tonal_width: float = 50,
    radius: int = 30,
    color_correction: float = 0,
    midtone_contrast: float = 0,
    black_clip: float = 0.01,
    white_clip: float = 0.01,
) -> Image:
    """
    Adjust shadows and highlights independently.

    Args:
        image: Input image
        shadows: Shadow adjustment (-100 to 100)
        highlights: Highlight adjustment (-100 to 100)
        shadow_tonal_width: Shadow range width (0-100)
        highlight_tonal_width: Highlight range width (0-100)
        radius: Radius for local tone mapping
        color_correction: Color adjustment (-100 to 100)
        midtone_contrast: Midtone contrast (-100 to 100)
        black_clip: Black clipping percentage
        white_clip: White clipping percentage

    Returns:
        Adjusted image
    """
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.pixel import PixelData

    if image.channels < 3:
        return image.copy()

    data = image.data.astype(np.float32)
    max_val = image.bit_depth.max_value

    # Normalize
    normalized = data[:, :, :3] / max_val

    # Calculate luminance
    luminance = (
        0.299 * normalized[:, :, 0]
        + 0.587 * normalized[:, :, 1]
        + 0.114 * normalized[:, :, 2]
    )

    # Create blurred luminance for local adjustment
    if radius > 0:
        ksize = radius * 2 + 1
        local_lum = cv2.GaussianBlur(luminance, (ksize, ksize), 0)
    else:
        local_lum = luminance

    # Shadow mask
    shadow_width = shadow_tonal_width / 100
    shadow_mask = np.clip(1 - local_lum / shadow_width, 0, 1) ** 2

    # Highlight mask
    highlight_width = highlight_tonal_width / 100
    highlight_mask = (
        np.clip((local_lum - (1 - highlight_width)) / highlight_width, 0, 1)
        ** 2
    )

    # Calculate adjustments
    shadow_adj = shadows / 100
    highlight_adj = highlights / 100

    # Apply to luminance
    adjusted_lum = luminance.copy()

    # Shadows: brighten or darken
    if shadow_adj > 0:
        # Brighten shadows
        adjusted_lum = adjusted_lum + shadow_mask * shadow_adj * (
            1 - adjusted_lum
        )
    else:
        # Darken shadows
        adjusted_lum = adjusted_lum * (1 + shadow_adj * shadow_mask)

    # Highlights: brighten or darken
    if highlight_adj > 0:
        # Brighten highlights
        adjusted_lum = adjusted_lum + highlight_mask * highlight_adj * (
            1 - adjusted_lum
        )
    else:
        # Darken highlights
        adjusted_lum = (
            adjusted_lum + highlight_mask * highlight_adj * adjusted_lum
        )

    # Midtone contrast
    if midtone_contrast != 0:
        mid_mask = 1 - shadow_mask - highlight_mask
        mid_mask = np.clip(mid_mask, 0, 1)

        contrast_factor = 1 + midtone_contrast / 100
        adjusted_lum = adjusted_lum + mid_mask * (adjusted_lum - 0.5) * (
            contrast_factor - 1
        )

    adjusted_lum = np.clip(adjusted_lum, 0, 1)

    # Apply luminance change to RGB
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(luminance > 0, adjusted_lum / luminance, 1)

    result = normalized * ratio[:, :, np.newaxis]

    # Color correction
    if color_correction != 0:
        # Adjust saturation in affected areas
        correction_factor = 1 + color_correction / 100

        # Calculate current saturation
        max_c = np.max(result, axis=2)
        min_c = np.min(result, axis=2)
        np.where(max_c > 0, (max_c - min_c) / max_c, 0)

        # Adjust
        mean_color = np.mean(result, axis=2, keepdims=True)
        result = mean_color + (result - mean_color) * correction_factor

    # Clip
    result = np.clip(result, black_clip, 1 - white_clip)

    # Renormalize
    result = (result - black_clip) / (1 - black_clip - white_clip)
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
