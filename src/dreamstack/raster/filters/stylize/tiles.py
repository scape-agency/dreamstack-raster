# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Tiles
=========================

Tiles effect implementation.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.image import Image


def tiles(
    image: Image,
    tile_count: int = 10,
    offset: int = 5,
    fill: str = "background",
) -> Image:
    """
    Apply tiles effect (breaks image into tiles).

    Args:
        image: Input image
        tile_count: Number of tiles
        offset: Maximum tile offset
        fill: Gap fill mode ('background', 'foreground', 'inverse')

    Returns:
        Tiled image
    """
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.pixel import PixelData

    data = image.data
    h, w = data.shape[:2]
    max_val = 255 if image.bit_depth.name == "UINT8" else 65535

    tile_h = h // tile_count
    tile_w = w // tile_count

    # Create background
    if fill == "background":
        result = np.zeros_like(data)
    elif fill == "foreground":
        result = np.ones_like(data) * max_val
    else:  # inverse
        result = max_val - data

    # Place tiles with random offsets
    for ty in range(tile_count):
        for tx in range(tile_count):
            # Source position
            sy = ty * tile_h
            sx = tx * tile_w

            # Random offset
            dy = np.random.randint(-offset, offset + 1)
            dx = np.random.randint(-offset, offset + 1)

            # Destination position
            dy_pos = sy + dy
            dx_pos = sx + dx

            # Clip
            src_y1 = sy
            src_y2 = min(sy + tile_h, h)
            src_x1 = sx
            src_x2 = min(sx + tile_w, w)

            dst_y1 = max(0, dy_pos)
            dst_y2 = min(h, dy_pos + tile_h)
            dst_x1 = max(0, dx_pos)
            dst_x2 = min(w, dx_pos + tile_w)

            # Adjust source based on clipping
            if dy_pos < 0:
                src_y1 -= dy_pos
            if dx_pos < 0:
                src_x1 -= dx_pos

            # Copy tile
            copy_h = min(dst_y2 - dst_y1, src_y2 - src_y1)
            copy_w = min(dst_x2 - dst_x1, src_x2 - src_x1)

            if copy_h > 0 and copy_w > 0:
                result[dst_y1 : dst_y1 + copy_h, dst_x1 : dst_x1 + copy_w] = (
                    data[src_y1 : src_y1 + copy_h, src_x1 : src_x1 + copy_w]
                )

    result_image = image.copy()
    result_image._pixel_data = PixelData(  # pylint: disable=protected-access
        data=result, pixel_format=image.pixel_format, bit_depth=image.bit_depth
    )

    return result_image
