# -*- coding: utf-8 -*-

"""
Dreamstack Raster - Twirl Distortion
====================================

Twirl distortion filter implementation.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import cv2
import numpy as np

if TYPE_CHECKING:
    from dreamstack.raster.core.image import Image


def twirl(
    image: Image,
    angle: float = 180,
    radius: Optional[float] = None,
    center: Optional[Tuple[float, float]] = None,
) -> Image:
    """
    Apply twirl distortion.

    Args:
        image: Input image
        angle: Maximum rotation angle in degrees
        radius: Effect radius (default: half image diagonal)
        center: Twirl center (relative 0-1)

    Returns:
        Distorted image
    """
    from dreamstack.raster.core.pixel import PixelData

    data = image.data
    h, w = data.shape[:2]

    if center is None:
        cx, cy = w / 2, h / 2
    else:
        cx, cy = center[0] * w, center[1] * h

    if radius is None:
        radius = np.sqrt(w**2 + h**2) / 2

    angle_rad = np.radians(angle)

    y, x = np.mgrid[:h, :w].astype(np.float32)

    dx = x - cx
    dy = y - cy
    distance = np.sqrt(dx**2 + dy**2)

    # Calculate rotation amount based on distance
    rotation = angle_rad * (1 - distance / radius)
    rotation = np.where(distance < radius, rotation, 0)

    # Apply rotation
    cos_r = np.cos(rotation)
    sin_r = np.sin(rotation)

    map_x = cx + cos_r * dx - sin_r * dy
    map_y = cy + sin_r * dx + cos_r * dy

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
