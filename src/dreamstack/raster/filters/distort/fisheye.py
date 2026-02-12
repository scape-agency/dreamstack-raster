"""
Dreamstack Raster - Fisheye Distortion
======================================

Fisheye lens distortion filter implementation.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from dreamstack.raster.core.image import Image


def fisheye(
    image: Image,
    amount: float = 100,
    center: tuple[float, float] | None = None,
) -> Image:
    """
    Apply fisheye lens distortion.

    Args:
        image: Input image
        amount: Effect strength (0-100)
        center: Effect center (relative 0-1)

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

    y, x = np.mgrid[:h, :w].astype(np.float32)

    # Normalize
    nx = (x - cx) / (w / 2)
    ny = (y - cy) / (h / 2)

    r = np.sqrt(nx**2 + ny**2)
    theta = np.arctan2(ny, nx)

    # Fisheye mapping
    amount_factor = amount / 100

    with np.errstate(divide="ignore", invalid="ignore"):
        # Apply barrel distortion
        new_r = r * (1 + amount_factor * r**2)

    new_nx = new_r * np.cos(theta)
    new_ny = new_r * np.sin(theta)

    map_x = cx + new_nx * (w / 2)
    map_y = cy + new_ny * (h / 2)

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
