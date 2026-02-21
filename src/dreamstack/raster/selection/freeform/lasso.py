"""
Lasso Selection
===============

Freehand lasso selection tool.

"""

# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    pass

from dreamstack.raster.selection.shapes.selection import Selection


def lasso(
    image_shape: tuple[int, int],
    points: list[tuple[int, int]],
    *,
    feather: int = 0,
    anti_alias: bool = True,
) -> Selection:
    """Create a freehand lasso selection from points.

    The selection is created by connecting the points
    and filling the resulting polygon.

    Args:
        image_shape: Shape of the image (height, width).
        points: List of (x, y) points defining the lasso path.
        feather: Feather radius for soft edges.
        anti_alias: Enable anti-aliasing on edges.

    Returns:
        Selection from lasso path.

    Example:
        >>> points = [(100, 100), (200, 100), (200, 200), (100, 200)]
        >>> sel = lasso((1080, 1920), points)
    """
    if len(points) < 3:
        h, w = image_shape
        return Selection(mask=np.zeros((h, w), dtype=np.uint8))

    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)

    # Convert to numpy array for cv2
    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))

    # Draw filled polygon
    line_type = cv2.LINE_AA if anti_alias else cv2.LINE_8
    cv2.fillPoly(mask, [pts], 255, line_type)

    if feather > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), feather)

    return Selection(mask=mask)
