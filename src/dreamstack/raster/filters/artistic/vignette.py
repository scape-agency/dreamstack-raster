"""
Dreamstack Raster - Vignette
============================

Vignette effect implementation.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dreamstack.raster.core.image import Image


def vignette(
    image: Image,
    amount: float = 50,
    midpoint: float = 50,
    roundness: float = 0,
    feather: float = 50,
) -> Image:
    """
    Apply vignette effect.

    Args:
        image: Input image
        amount: Vignette darkness (-100 to 100)
        midpoint: Distance from center where vignette starts (0-100)
        roundness: Shape roundness (-100 to 100)
        feather: Edge feather amount (0-100)

    Returns:
        Vignetted image
    """
    from dreamstack.raster.core.pixel import PixelData

    data = image.data.astype(np.float32)
    h, w = data.shape[:2]
    max_val = 255 if image.bit_depth.name == "UINT8" else 65535

    # Create vignette mask
    y, x = np.ogrid[:h, :w]

    cx, cy = w / 2, h / 2

    # Adjust for roundness
    aspect = w / h
    if roundness >= 0:
        ax = 1.0
        ay = 1.0 / aspect
    else:
        ax = aspect
        ay = 1.0

    1 + roundness / 100

    # Distance from center (normalized)
    dx = (x - cx) / (w / 2) * ax
    dy = (y - cy) / (h / 2) * ay
    distance = np.sqrt(dx**2 + dy**2)

    # Apply midpoint
    midpoint_normalized = midpoint / 100

    # Apply feather
    feather_normalized = max(0.01, feather / 100)

    # Create mask
    mask = 1 - np.clip((distance - midpoint_normalized) / feather_normalized, 0, 1)

    # Apply amount
    amount_normalized = amount / 100
    if amount_normalized < 0:
        # Brighten edges
        multiplier = 1 + (1 - mask) * abs(amount_normalized)
    else:
        # Darken edges
        multiplier = 1 - (1 - mask) * amount_normalized

    # Apply to image
    if data.ndim == 3:
        multiplier = multiplier[:, :, np.newaxis]

    result = data * multiplier
    result = np.clip(result, 0, max_val)

    result_image = image.copy()
    result_image._pixel_data = PixelData(
        data=result.astype(image.data.dtype),
        pixel_format=image.pixel_format,
        bit_depth=image.bit_depth,
    )

    return result_image
