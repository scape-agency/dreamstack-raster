# -*- coding: utf-8 -*-

"""
Dreamstack Raster - Extrude
===========================

Extrude blocks effect implementation.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dreamstack.raster.core.image import Image


def extrude(
    image: Image, size: int = 10, depth: int = 30, direction: str = "random"
) -> Image:
    """
    Apply extrude blocks effect.

    Args:
        image: Input image
        size: Block size
        depth: Extrusion depth
        direction: 'random' or 'center'

    Returns:
        Extruded image
    """
    from dreamstack.raster.core.pixel import PixelData

    data = image.data
    h, w = data.shape[:2]

    result = data.copy()

    cx, cy = w // 2, h // 2

    for y in range(0, h, size):
        for x in range(0, w, size):
            block_h = min(size, h - y)
            block_w = min(size, w - x)

            # Get block average color
            block = data[y : y + block_h, x : x + block_w]
            avg_color = block.mean(axis=(0, 1))

            # Extrusion direction
            if direction == "random":
                dx = np.random.randint(-depth, depth + 1)
                dy = np.random.randint(-depth, depth + 1)
            else:  # center
                # Direction towards center
                dir_x = cx - (x + block_w // 2)
                dir_y = cy - (y + block_h // 2)
                length = np.sqrt(dir_x**2 + dir_y**2)
                if length > 0:
                    dx = int(depth * dir_x / length)
                    dy = int(depth * dir_y / length)
                else:
                    dx = dy = 0

            # Draw extrusion (simplified as solid color block)
            # This is a simplified version - full implementation would draw 3D projection
            result[y : y + block_h, x : x + block_w] = avg_color

    result_image = image.copy()
    result_image._pixel_data = PixelData(
        data=result, pixel_format=image.pixel_format, bit_depth=image.bit_depth
    )

    return result_image
