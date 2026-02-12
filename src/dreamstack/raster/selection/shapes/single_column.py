"""
Single Column Selection
=======================

Create single-pixel column selections.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

from dreamstack.raster.selection.shapes.selection import Selection


def single_column(
    image_shape: tuple[int, int],
    x: int,
) -> Selection:
    """Create a single-column selection.

    Selects exactly one vertical column of pixels.

    Args:
        image_shape: Shape of the image (height, width).
        x: X-coordinate of the column to select.

    Returns:
        Selection object with single-column mask.

    Example:
        >>> sel = single_column((1080, 1920), 960)  # Middle column
    """
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)

    if 0 <= x < w:
        mask[:, x] = 255

    return Selection(mask=mask, bounds=(x, 0, 1, h))
