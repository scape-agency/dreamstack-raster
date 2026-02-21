# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Rectangle
=========

Draw rectangles on images.

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


def rectangle(
    image: NDArray[np.uint8],
    x: int,
    y: int,
    width: int,
    height: int,
    color: tuple[int, int, int] | tuple[int, int, int, int] = (0, 0, 0, 255),
    *,
    thickness: int = 1,
    filled: bool = False,
) -> NDArray[np.uint8]:
    """Draw a rectangle on an image.

    Args:
        image: Image to draw on.
        x: Top-left x coordinate.
        y: Top-left y coordinate.
        width: Rectangle width.
        height: Rectangle height.
        color: Rectangle color (RGB or RGBA).
        thickness: Border thickness (ignored if filled).
        filled: If True, fill the rectangle.

    Returns:
        Image with rectangle drawn.

    Example:
        >>> # Draw outline
        >>> result = rectangle(image, 10, 10, 100, 50, (255, 0, 0))
        >>> # Draw filled
        >>> result = rectangle(image, 10, 10, 100, 50, (0, 255, 0), filled=True)
    """
    result = image.copy()

    # Convert color to BGR
    if len(color) == 4:
        bgr_color = (color[2], color[1], color[0])
    else:
        bgr_color = (color[2], color[1], color[0])

    pt1 = (x, y)
    pt2 = (x + width, y + height)

    thick = -1 if filled else thickness

    cv2.rectangle(result, pt1, pt2, bgr_color, thick)

    return result
