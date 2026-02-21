# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Lens Blur
=============================

Lens blur (bokeh effect) filter implementation.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.image import Image


def lens_blur(
    image: Image,
    radius: float = 10,
    blade_count: int = 6,
    rotation: float = 0,
    brightness: float = 0,
    threshold: float = 255,
) -> Image:
    """
    Apply lens blur (bokeh effect).

    Args:
        image: Input image
        radius: Blur radius
        blade_count: Number of aperture blades (0 for circular)
        rotation: Blade rotation in degrees
        brightness: Specular highlights brightness
        threshold: Threshold for specular highlights

    Returns:
        Blurred image
    """
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.pixel import PixelData

    # Create aperture kernel
    ksize = int(radius * 2) | 1
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    center = ksize // 2

    if blade_count == 0:
        # Circular aperture
        y, x = np.ogrid[:ksize, :ksize]
        mask = (x - center) ** 2 + (y - center) ** 2 <= radius**2
        kernel[mask] = 1
    else:
        # Polygonal aperture
        for y in range(ksize):
            for x in range(ksize):
                dx = x - center
                dy = y - center
                r = np.sqrt(dx**2 + dy**2)

                if r <= radius:
                    angle = np.arctan2(dy, dx) + np.radians(rotation)
                    # Check if inside polygon
                    blade_angle = 2 * np.pi / blade_count
                    sector = angle % blade_angle - blade_angle / 2
                    edge_dist = (
                        radius * np.cos(blade_angle / 2) / np.cos(sector)
                    )

                    if r <= edge_dist:
                        kernel[y, x] = 1

    if kernel.sum() > 0:
        kernel /= kernel.sum()

    data = image.data.astype(np.float32)
    max_val = 255 if image.bit_depth.name == "UINT8" else 65535

    # Apply specular highlights boost
    if brightness > 0:
        highlights = np.where(
            data > threshold, data * (1 + brightness / 100), data
        )
        data = highlights

    # Apply blur
    if data.ndim == 3:
        result = np.zeros_like(data)
        for i in range(data.shape[2]):
            result[:, :, i] = cv2.filter2D(data[:, :, i], -1, kernel)
    else:
        result = cv2.filter2D(data, -1, kernel)

    result = np.clip(result, 0, max_val)

    result_image = image.copy()
    result_image._pixel_data = PixelData(  # pylint: disable=protected-access
        data=result.astype(image.data.dtype),
        pixel_format=image.pixel_format,
        bit_depth=image.bit_depth,
    )

    return result_image
