# -*- coding: utf-8 -*-

"""
Dreamstack Raster - Sphere Distortion
=====================================

Spherize distortion filter implementation.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import cv2
import numpy as np

if TYPE_CHECKING:
    from dreamstack.raster.core.image import Image


def sphere(
    image: Image,
    amount: float = 100,
    center: Optional[Tuple[float, float]] = None,
) -> Image:
    """
    Apply spherize distortion.

    Args:
        image: Input image
        amount: Distortion amount (-100 to 100)
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

    # Normalize to -1 to 1
    nx = (x - cx) / (w / 2)
    ny = (y - cy) / (h / 2)

    r = np.sqrt(nx**2 + ny**2)

    # Spherize mapping
    amount_factor = amount / 100

    with np.errstate(divide="ignore", invalid="ignore"):
        # Inside unit circle
        mask = r <= 1

        if amount_factor >= 0:
            # Spherize outward (bulge)
            new_r = np.where(mask, np.arcsin(r) / (np.pi / 2), r)
        else:
            # Spherize inward (pinch)
            new_r = np.where(mask, np.sin(r * np.pi / 2), r)

        scale = np.where(r > 0, new_r / r, 1)
        scale = 1 + (scale - 1) * abs(amount_factor)

    map_x = cx + (x - cx) * scale
    map_y = cy + (y - cy) * scale

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
