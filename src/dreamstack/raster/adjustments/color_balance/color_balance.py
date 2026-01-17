# -*- coding: utf-8 -*-

"""Color balance adjustment function."""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np

if TYPE_CHECKING:
    from dreamstack.raster.core.image import Image


def color_balance(
    image: Image,
    shadows: Tuple[float, float, float] = (0, 0, 0),
    midtones: Tuple[float, float, float] = (0, 0, 0),
    highlights: Tuple[float, float, float] = (0, 0, 0),
    preserve_luminosity: bool = True,
) -> Image:
    """
    Adjust color balance for shadows, midtones, and highlights.

    Args:
        image: Input image
        shadows: (cyan-red, magenta-green, yellow-blue) for shadows (-100 to 100)
        midtones: (cyan-red, magenta-green, yellow-blue) for midtones
        highlights: (cyan-red, magenta-green, yellow-blue) for highlights
        preserve_luminosity: Maintain luminosity after adjustment

    Returns:
        Adjusted image
    """
    from dreamstack.raster.core.pixel import PixelData

    if image.channels < 3:
        return image.copy()

    data = image.data.astype(np.float32)
    max_val = 255 if image.bit_depth.name == "UINT8" else 65535

    # Normalize to 0-1
    normalized = data[:, :, :3] / max_val

    # Calculate luminosity weights
    luminosity = (
        0.299 * normalized[:, :, 0]
        + 0.587 * normalized[:, :, 1]
        + 0.114 * normalized[:, :, 2]
    )

    # Define tonal ranges
    # Shadows: 0-0.33, Midtones: 0.33-0.67, Highlights: 0.67-1
    shadow_mask = np.clip(1 - luminosity / 0.33, 0, 1) ** 2
    highlight_mask = np.clip((luminosity - 0.67) / 0.33, 0, 1) ** 2
    midtone_mask = 1 - shadow_mask - highlight_mask
    midtone_mask = np.clip(midtone_mask, 0, 1)

    result = normalized.copy()

    # Apply adjustments
    for i, (s, m, h) in enumerate(zip(shadows, midtones, highlights)):
        adjustment = (
            shadow_mask * s + midtone_mask * m + highlight_mask * h
        ) / 100
        result[:, :, i] = result[:, :, i] + adjustment

    result = np.clip(result, 0, 1)

    # Preserve luminosity
    if preserve_luminosity:
        new_luminosity = (
            0.299 * result[:, :, 0]
            + 0.587 * result[:, :, 1]
            + 0.114 * result[:, :, 2]
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(
                new_luminosity > 0, luminosity / new_luminosity, 1
            )

        for i in range(3):
            result[:, :, i] = result[:, :, i] * ratio

        result = np.clip(result, 0, 1)

    # Scale back
    final = data.copy()
    final[:, :, :3] = result * max_val

    result_image = image.copy()
    result_image._pixel_data = PixelData(
        data=final.astype(image.data.dtype),
        pixel_format=image.pixel_format,
        bit_depth=image.bit_depth,
    )

    return result_image
