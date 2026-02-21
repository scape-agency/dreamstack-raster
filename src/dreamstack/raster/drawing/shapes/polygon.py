# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Polygon
=======

Draw polygons on images.

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
    # pylint: disable=import-outside-toplevel
    from numpy.typing import NDArray


def polygon(
    image: NDArray[np.uint8],
    points: list[tuple[int, int]],
    color: tuple[int, int, int] | tuple[int, int, int, int] = (0, 0, 0, 255),
    *,
    thickness: int = 1,
    filled: bool = False,
    anti_alias: bool = True,
) -> NDArray[np.uint8]:
    """Draw a polygon on an image.

    Args:
        image: Image to draw on.
        points: List of (x, y) vertices.
        color: Polygon color (RGB or RGBA).
        thickness: Border thickness (ignored if filled).
        filled: If True, fill the polygon.
        anti_alias: If True, use anti-aliased drawing.

    Returns:
        Image with polygon drawn.

    Example:
        >>> # Draw triangle
        >>> pts = [(100, 10), (50, 100), (150, 100)]
        >>> result = polygon(image, pts, (255, 0, 0), filled=True)
    """
    if len(points) < 3:
        return image.copy()

    result = image.copy()

    # Convert color to BGR
    if len(color) == 4:
        bgr_color = (color[2], color[1], color[0])
    else:
        bgr_color = (color[2], color[1], color[0])

    # Convert points to numpy array
    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))

    line_type = cv2.LINE_AA if anti_alias else cv2.LINE_8

    if filled:
        cv2.fillPoly(result, [pts], bgr_color, line_type)
    else:
        cv2.polylines(result, [pts], True, bgr_color, thickness, line_type)

    return result
