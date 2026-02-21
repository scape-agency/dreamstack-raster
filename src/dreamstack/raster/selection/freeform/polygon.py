"""
Polygon Selection
=================

Create selections from polygon vertices.

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


def polygon(
    image_shape: tuple[int, int],
    vertices: list[tuple[int, int]],
    *,
    feather: int = 0,
    anti_alias: bool = True,
) -> Selection:
    """Create a polygon selection from vertices.

    Unlike lasso, polygon creates straight-line connections
    between vertices for precise geometric selections.

    Args:
        image_shape: Shape of the image (height, width).
        vertices: List of (x, y) vertices defining the polygon.
        feather: Feather radius for soft edges.
        anti_alias: Enable anti-aliasing on edges.

    Returns:
        Selection from polygon.

    Example:
        >>> # Triangle selection
        >>> vertices = [(500, 100), (900, 400), (100, 400)]
        >>> sel = polygon((1080, 1920), vertices)
    """
    if len(vertices) < 3:
        h, w = image_shape
        return Selection(mask=np.zeros((h, w), dtype=np.uint8))

    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)

    # Convert to numpy array
    pts = np.array(vertices, dtype=np.int32).reshape((-1, 1, 2))

    # Draw filled polygon
    line_type = cv2.LINE_AA if anti_alias else cv2.LINE_8
    cv2.fillPoly(mask, [pts], 255, line_type)

    if feather > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), feather)

    return Selection(mask=mask)
