"""
Single Row Selection
====================

Create single-pixel row selections.

"""

# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

# =============================================================================
# Type Checking Imports
# =============================================================================

if TYPE_CHECKING:
    pass

from dreamstack.raster.selection.shapes.selection import Selection


def single_row(
    image_shape: tuple[int, int],
    y: int,
) -> Selection:
    """Create a single-row selection.

    Selects exactly one horizontal row of pixels.

    Args:
        image_shape: Shape of the image (height, width).
        y: Y-coordinate of the row to select.

    Returns:
        Selection object with single-row mask.

    Example:
        >>> sel = single_row((1080, 1920), 540)  # Middle row
    """
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)

    if 0 <= y < h:
        mask[y, :] = 255

    return Selection(mask=mask, bounds=(0, y, w, 1))
