# -*- coding: utf-8 -*-

"""
Dreamstack Raster - Stipple
===========================

Stipple effect implementation.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from dreamstack.raster.core.image import Image


def stipple(image: Image, density: float = 0.3, dot_size: int = 2) -> Image:
    """
    Apply stipple effect (random dots based on intensity).

    Args:
        image: Input image
        density: Dot density (0-1)
        dot_size: Size of dots

    Returns:
        Stippled image
    """
    from dreamstack.raster.core.pixel import PixelData

    gray = image.to_grayscale().data.astype(np.float32)
    if gray.ndim == 3:
        gray = gray[:, :, 0]

    max_val = 255 if image.bit_depth.name == "UINT8" else 65535

    h, w = gray.shape
    result = np.ones((h, w), dtype=np.float32) * max_val

    # Invert intensity for dot probability
    prob = 1 - (gray / max_val)

    # Random dots
    random = np.random.random((h, w))
    dots = random < (prob * density)

    # Enlarge dots
    if dot_size > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dot_size, dot_size)
        )
        dots = cv2.dilate(dots.astype(np.uint8), kernel).astype(bool)

    result[dots] = 0

    result_image = image.copy()
    result_image._pixel_data = PixelData(
        data=result[:, :, np.newaxis].astype(image.data.dtype),
        pixel_format=image.to_grayscale().pixel_format,
        bit_depth=image.bit_depth,
    )

    return result_image
