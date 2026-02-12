"""
Elliptical Selection
====================

Create elliptical selections.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    pass

from dreamstack.raster.selection.shapes.selection import Selection


def elliptical(
    image_shape: tuple[int, int],
    center_x: int,
    center_y: int,
    radius_x: int,
    radius_y: int,
    *,
    feather: int = 0,
    anti_alias: bool = True,
) -> Selection:
    """Create an elliptical selection.

    Args:
        image_shape: Shape of the image (height, width).
        center_x: Center x-coordinate.
        center_y: Center y-coordinate.
        radius_x: Horizontal radius.
        radius_y: Vertical radius.
        feather: Feather radius for soft edges.
        anti_alias: Enable anti-aliasing on edges.

    Returns:
        Selection object with elliptical mask.

    Example:
        >>> # Circular selection
        >>> sel = elliptical((1080, 1920), 960, 540, 200, 200)
        >>> # Elliptical selection
        >>> sel = elliptical((1080, 1920), 500, 300, 150, 100)
    """
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)

    # Draw filled ellipse
    line_type = cv2.LINE_AA if anti_alias else cv2.LINE_8
    cv2.ellipse(
        mask,
        (center_x, center_y),
        (radius_x, radius_y),
        0,  # angle
        0,
        360,  # start/end angle
        255,  # color
        -1,  # filled
        line_type,
    )

    if feather > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), feather)

    # Calculate bounds
    x1 = max(0, center_x - radius_x)
    y1 = max(0, center_y - radius_y)
    x2 = min(w, center_x + radius_x)
    y2 = min(h, center_y + radius_y)

    return Selection(mask=mask, bounds=(x1, y1, x2 - x1, y2 - y1))
