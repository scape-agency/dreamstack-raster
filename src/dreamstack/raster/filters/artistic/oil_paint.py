# -*- coding: utf-8 -*-

"""
Dreamstack Raster - Oil Paint
=============================

Oil painting effect implementation.

"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from dreamstack.raster.core.image import Image


def oil_paint(image: Image, brush_size: int = 6, roughness: int = 1) -> Image:
    """
    Apply oil painting effect.

    Args:
        image: Input image
        brush_size: Size of brush (1-8)
        roughness: Color intensity levels

    Returns:
        Stylized image
    """
    from dreamstack.raster.core.pixel import PixelData

    data = image.data

    # Convert to 8-bit for processing
    if image.bit_depth.name != "UINT8":
        data = (data / data.max() * 255).astype(np.uint8)

    if data.ndim == 3 and data.shape[2] >= 3:
        # OpenCV xphoto module has oil painting
        try:
            result = cv2.xphoto.oilPainting(
                data[:, :, :3], brush_size, roughness
            )
            if data.shape[2] == 4:
                result = np.dstack([result, data[:, :, 3]])
        except AttributeError:
            # Fallback implementation
            result = _oil_paint_fallback(data, brush_size, roughness)
    else:
        result = _oil_paint_fallback(data, brush_size, roughness)

    # Convert back
    if image.bit_depth.name != "UINT8":
        max_val = 65535 if image.bit_depth.name == "UINT16" else 1.0
        result = (result / 255.0 * max_val).astype(image.data.dtype)

    result_image = image.copy()
    result_image._pixel_data = PixelData(
        data=result, pixel_format=image.pixel_format, bit_depth=image.bit_depth
    )

    return result_image


def _oil_paint_fallback(
    data: np.ndarray, brush_size: int, roughness: int
) -> np.ndarray:
    """Fallback oil painting implementation."""
    h, w = data.shape[:2]
    result = np.zeros_like(data)

    # Quantize colors
    levels = max(1, 256 // (roughness + 1))
    quantized = (data // levels) * levels

    radius = brush_size

    for y in range(h):
        for x in range(w):
            y_min = max(0, y - radius)
            y_max = min(h, y + radius + 1)
            x_min = max(0, x - radius)
            x_max = min(w, x + radius + 1)

            region = quantized[y_min:y_max, x_min:x_max]

            if data.ndim == 3:
                # Find most common color
                flat = region.reshape(-1, region.shape[-1])
                tuples = [tuple(c) for c in flat]
                most_common = Counter(tuples).most_common(1)[0][0]
                result[y, x] = most_common
            else:
                flat = region.flatten()
                most_common = Counter(flat).most_common(1)[0][0]
                result[y, x] = most_common

    return result
