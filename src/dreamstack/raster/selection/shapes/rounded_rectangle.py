"""
Rounded Rectangle Selection
============================

Create rounded rectangle selections.

"""

# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

# =============================================================================
# Type Checking Imports
# =============================================================================

if TYPE_CHECKING:
    pass

from dreamstack.raster.selection.shapes.selection import Selection


def rounded_rectangle(
    image_shape: tuple[int, int],
    x: int,
    y: int,
    width: int,
    height: int,
    radius: int,
    *,
    feather: int = 0,
) -> Selection:
    """Create a rounded rectangle selection.

    Args:
        image_shape: Shape of the image (height, width).
        x: Left edge x-coordinate.
        y: Top edge y-coordinate.
        width: Width of rectangle.
        height: Height of rectangle.
        radius: Corner radius.
        feather: Feather radius for soft edges.

    Returns:
        Selection object with rounded rectangle mask.

    Example:
        >>> sel = rounded_rectangle((1080, 1920), 100, 100, 400, 300, 20)
    """
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)

    # Clamp radius
    radius = min(radius, width // 2, height // 2)

    # Draw rounded rectangle by combining shapes
    x1, y1 = x, y
    x2, y2 = x + width, y + height

    # Main rectangle (without corners)
    cv2.rectangle(mask, (x1 + radius, y1), (x2 - radius, y2), 255, -1)
    cv2.rectangle(mask, (x1, y1 + radius), (x2, y2 - radius), 255, -1)

    # Corner circles
    cv2.circle(mask, (x1 + radius, y1 + radius), radius, 255, -1, cv2.LINE_AA)
    cv2.circle(mask, (x2 - radius, y1 + radius), radius, 255, -1, cv2.LINE_AA)
    cv2.circle(mask, (x1 + radius, y2 - radius), radius, 255, -1, cv2.LINE_AA)
    cv2.circle(mask, (x2 - radius, y2 - radius), radius, 255, -1, cv2.LINE_AA)

    if feather > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), feather)

    return Selection(
        mask=mask,
        bounds=(
            max(0, x1),
            max(0, y1),
            min(width, w - x1),
            min(height, h - y1),
        ),
    )
