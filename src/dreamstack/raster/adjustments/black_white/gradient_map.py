# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Gradient map function."""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dreamstack.raster.core.image import Image


def gradient_map(
    image: Image,
    gradient: list[tuple[float, tuple[int, int, int]]],
    reverse: bool = False,
    dither: bool = False,
) -> Image:
    """
    Map luminance to a color gradient.

    Args:
        image: Input image
        gradient: List of (position, (r, g, b)) tuples
                  Position is 0-1, colors are 0-255
        reverse: Reverse the gradient
        dither: Apply dithering

    Returns:
        Gradient-mapped image
    """
    from dreamstack.raster.core.pixel import PixelData

    if image.channels < 3:
        return image.copy()

    data = image.data.astype(np.float32)
    max_val = 255 if image.bit_depth.name == "UINT8" else 65535

    normalized = data[:, :, :3] / max_val

    # Calculate luminance
    luminance = (
        0.299 * normalized[:, :, 0]
        + 0.587 * normalized[:, :, 1]
        + 0.114 * normalized[:, :, 2]
    )

    if reverse:
        luminance = 1 - luminance

    # Sort gradient by position
    gradient = sorted(gradient, key=lambda x: x[0])

    # Ensure we have endpoints
    if gradient[0][0] > 0:
        gradient.insert(0, (0, gradient[0][1]))
    if gradient[-1][0] < 1:
        gradient.append((1, gradient[-1][1]))

    # Build lookup table
    lut_size = 256
    lut = np.zeros((lut_size, 3))

    for i in range(lut_size):
        pos = i / (lut_size - 1)

        # Find surrounding gradient stops
        for j in range(len(gradient) - 1):
            if gradient[j][0] <= pos <= gradient[j + 1][0]:
                # Interpolate
                t = (
                    (pos - gradient[j][0])
                    / (gradient[j + 1][0] - gradient[j][0])
                    if gradient[j + 1][0] != gradient[j][0]
                    else 0
                )

                for c in range(3):
                    lut[i, c] = (
                        gradient[j][1][c] * (1 - t) + gradient[j + 1][1][c] * t
                    )
                break

    # Normalize LUT
    lut = lut / 255

    # Apply gradient map
    indices = (luminance * (lut_size - 1)).astype(int)
    indices = np.clip(indices, 0, lut_size - 1)

    result = np.zeros((*luminance.shape, 3))
    for c in range(3):
        result[:, :, c] = lut[indices, c]

    # Add dithering
    if dither:
        noise = np.random.random(luminance.shape) * (
            1 / (lut_size - 1)
        ) - 0.5 / (lut_size - 1)
        result = result + noise[:, :, np.newaxis]

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
