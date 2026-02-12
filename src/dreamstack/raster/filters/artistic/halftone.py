"""
Dreamstack Raster - Halftone
============================

Halftone dot pattern effect implementation.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dreamstack.raster.core.image import Image


def halftone(
    image: Image, dot_size: int = 4, angle: float = 45, grayscale: bool = True
) -> Image:
    """
    Apply halftone dot pattern.

    Args:
        image: Input image
        dot_size: Size of halftone dots
        angle: Pattern angle in degrees
        grayscale: Convert to grayscale first

    Returns:
        Halftone image
    """
    from dreamstack.raster.core.pixel import PixelData

    data = image.data.astype(np.float32)
    max_val = 255 if image.bit_depth.name == "UINT8" else 65535

    if grayscale or image.channels == 1:
        gray = image.to_grayscale().data.astype(np.float32)
        if gray.ndim == 3:
            gray = gray[:, :, 0]
        result = _create_halftone(gray, dot_size, angle, max_val)
        result = result[:, :, np.newaxis]
    else:
        # CMYK-style halftone for color
        result = np.zeros_like(data)
        angles = [angle, angle + 15, angle + 30, angle + 45]

        for i in range(min(data.shape[2], 4)):
            channel = data[:, :, i]
            ht = _create_halftone(channel, dot_size, angles[i % 4], max_val)
            result[:, :, i] = ht

    result_image = image.copy()
    result_image._pixel_data = PixelData(
        data=result.astype(image.data.dtype),
        pixel_format=image.pixel_format,
        bit_depth=image.bit_depth,
    )

    return result_image


def _create_halftone(
    data: np.ndarray, dot_size: int, angle: float, max_val: float
) -> np.ndarray:
    """Create halftone pattern for single channel."""
    h, w = data.shape
    result = np.zeros_like(data)

    # Create dot centers
    step = dot_size * 2

    for y in range(0, h, step):
        for x in range(0, w, step):
            # Get average intensity in block
            y2 = min(y + step, h)
            x2 = min(x + step, w)
            block = data[y:y2, x:x2]
            intensity = block.mean() / max_val

            # Draw dot with radius based on intensity
            radius = int((1 - intensity) * dot_size)

            if radius > 0:
                cy = y + step // 2
                cx = x + step // 2

                yy, xx = np.ogrid[:h, :w]
                mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
                result[mask] = max_val

    return result
