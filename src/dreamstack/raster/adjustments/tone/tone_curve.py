"""Parametric tone curve function."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dreamstack.raster.core.image import Image


def tone_curve(
    image: Image,
    shadows: float = 0,
    darks: float = 0,
    lights: float = 0,
    highlights: float = 0,
) -> Image:
    """
    Apply parametric tone curve.

    Args:
        image: Input image
        shadows: Shadow adjustment (-100 to 100)
        darks: Dark midtone adjustment (-100 to 100)
        lights: Light midtone adjustment (-100 to 100)
        highlights: Highlight adjustment (-100 to 100)

    Returns:
        Adjusted image
    """
    from dreamstack.raster.core.pixel import PixelData

    data = image.data.astype(np.float32)
    max_val = 255 if image.bit_depth.name == "UINT8" else 65535

    # Build parametric curve
    # Zones: 0-25% shadows, 25-50% darks, 50-75% lights, 75-100% highlights

    x = np.linspace(0, 1, 256)
    y = x.copy()

    # Shadow zone (0-25%)
    shadow_mask = np.clip(1 - x / 0.25, 0, 1) ** 2
    y = y + shadow_mask * shadows / 100 * 0.25

    # Darks zone (25-50%)
    darks_mask = 1 - np.abs(x - 0.375) / 0.125
    darks_mask = np.clip(darks_mask, 0, 1) ** 2
    y = y + darks_mask * darks / 100 * 0.25

    # Lights zone (50-75%)
    lights_mask = 1 - np.abs(x - 0.625) / 0.125
    lights_mask = np.clip(lights_mask, 0, 1) ** 2
    y = y + lights_mask * lights / 100 * 0.25

    # Highlights zone (75-100%)
    highlight_mask = np.clip((x - 0.75) / 0.25, 0, 1) ** 2
    y = y + highlight_mask * highlights / 100 * 0.25

    y = np.clip(y, 0, 1)

    # Create lookup table
    lut = (y * 255).astype(np.uint8)

    # Normalize to 0-255 for lookup
    normalized = (data / max_val * 255).astype(np.uint8)

    result = np.zeros_like(data)

    if data.ndim == 3:
        for i in range(data.shape[2]):
            result[:, :, i] = lut[normalized[:, :, i]]
    else:
        result = lut[normalized]

    result = result / 255 * max_val

    result_image = image.copy()
    result_image._pixel_data = PixelData(
        data=result.astype(image.data.dtype),
        pixel_format=image.pixel_format,
        bit_depth=image.bit_depth,
    )

    return result_image
