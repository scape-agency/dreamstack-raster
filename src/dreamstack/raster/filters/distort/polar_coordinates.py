# -*- coding: utf-8 -*-

"""
Dreamstack Raster - Polar Coordinates
=====================================

Polar/rectangular coordinate transformation filter implementation.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import cv2
import numpy as np

if TYPE_CHECKING:
    from dreamstack.raster.core.image import Image


def polar_coordinates(
    image: Image,
    mode: str = "rectangular_to_polar",
    center: Optional[Tuple[float, float]] = None,
) -> Image:
    """
    Convert between polar and rectangular coordinates.

    Args:
        image: Input image
        mode: 'rectangular_to_polar' or 'polar_to_rectangular'
        center: Transform center (relative 0-1)

    Returns:
        Transformed image
    """
    from dreamstack.raster.core.pixel import PixelData

    data = image.data
    h, w = data.shape[:2]

    if center is None:
        cx, cy = w / 2, h / 2
    else:
        cx, cy = center[0] * w, center[1] * h

    max_radius = np.sqrt(cx**2 + cy**2)

    y, x = np.mgrid[:h, :w].astype(np.float32)

    if mode == "rectangular_to_polar":
        # Map x to angle, y to radius
        angle = (x / w) * 2 * np.pi
        radius = (y / h) * max_radius

        map_x = cx + radius * np.cos(angle)
        map_y = cy + radius * np.sin(angle)

    else:  # polar_to_rectangular
        dx = x - cx
        dy = y - cy

        radius = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx)

        # Normalize angle to 0-2pi
        angle = (angle + np.pi) % (2 * np.pi)

        map_x = (angle / (2 * np.pi)) * w
        map_y = (radius / max_radius) * h

    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    result = cv2.remap(
        data, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
    )

    result_image = image.copy()
    result_image._pixel_data = PixelData(
        data=result, pixel_format=image.pixel_format, bit_depth=image.bit_depth
    )

    return result_image
