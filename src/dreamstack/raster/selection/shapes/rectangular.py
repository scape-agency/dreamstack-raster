"""
Rectangular Selection
=====================

Create rectangular selections.

"""

# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from dreamstack.raster.selection.shapes.selection import Selection

# =============================================================================
# Type Checking Imports
# =============================================================================

if TYPE_CHECKING:
    pass


def rectangular(
    image_shape: tuple[int, int],
    x: int,
    y: int,
    width: int,
    height: int,
    *,
    feather: int = 0,
) -> Selection:
    """Create a rectangular selection.

    Args:
        image_shape: Shape of the image (height, width).
        x: Left edge x-coordinate.
        y: Top edge y-coordinate.
        width: Width of rectangle.
        height: Height of rectangle.
        feather: Feather radius for soft edges.

    Returns:
        Selection object with rectangular mask.

    Example:
        >>> sel = rectangular((1080, 1920), 100, 100, 400, 300)
        >>> masked = sel.apply_to_image(image)
    """
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)

    # Clamp coordinates
    x1 = max(0, min(x, w - 1))
    y1 = max(0, min(y, h - 1))
    x2 = max(0, min(x + width, w))
    y2 = max(0, min(y + height, h))

    mask[y1:y2, x1:x2] = 255

    if feather > 0:
        import cv2

        mask = cv2.GaussianBlur(mask, (0, 0), feather)

    return Selection(mask=mask, bounds=(x1, y1, x2 - x1, y2 - y1))
