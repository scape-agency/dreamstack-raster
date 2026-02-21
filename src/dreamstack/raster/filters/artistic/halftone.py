# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Halftone
============================

Halftone dot pattern effect implementation.

"""


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
    # pylint: disable=import-outside-toplevel
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
    result_image._pixel_data = PixelData(  # pylint: disable=protected-access
        data=result.astype(image.data.dtype),
        pixel_format=image.pixel_format,
        bit_depth=image.bit_depth,
    )

    return result_image


def _create_halftone(
    data: np.ndarray,
    dot_size: int,
    angle: float,
    max_val: float,
) -> np.ndarray:
    """Create halftone pattern for single channel with rotation."""
    h, w = data.shape
    result = np.zeros_like(data)

    # Create dot centers
    step = dot_size * 2

    # Convert angle to radians
    angle_rad = np.radians(angle)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    # Calculate rotated grid bounds
    diag = int(np.sqrt(h * h + w * w))
    cx_img, cy_img = w // 2, h // 2

    # Iterate over rotated grid
    for gy in range(-diag // step, diag // step + 1):
        for gx in range(-diag // step, diag // step + 1):
            # Grid position in rotated space
            rx = gx * step
            ry = gy * step

            # Transform back to image space
            cx = int(cx_img + rx * cos_a - ry * sin_a)
            cy = int(cy_img + rx * sin_a + ry * cos_a)

            # Skip if outside image
            if cx < 0 or cx >= w or cy < 0 or cy >= h:
                continue

            # Get average intensity in block around center
            y1 = max(0, cy - step // 2)
            y2 = min(h, cy + step // 2)
            x1 = max(0, cx - step // 2)
            x2 = min(w, cx + step // 2)

            if y2 > y1 and x2 > x1:
                block = data[y1:y2, x1:x2]
                intensity = block.mean() / max_val

                # Draw dot with radius based on intensity
                radius = int((1 - intensity) * dot_size)

                if radius > 0:
                    yy, xx = np.ogrid[:h, :w]
                    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
                    result[mask] = max_val

    return result
