"""
Arrow
=====

Draw arrows on images.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def arrow(
    image: NDArray[np.uint8],
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int] | tuple[int, int, int, int] = (0, 0, 0, 255),
    *,
    thickness: int = 2,
    tip_length: float = 0.1,
    anti_alias: bool = True,
) -> NDArray[np.uint8]:
    """Draw an arrow on an image.

    Args:
        image: Image to draw on.
        start: Arrow start point (x, y).
        end: Arrow end point (tip) (x, y).
        color: Arrow color (RGB or RGBA).
        thickness: Line thickness in pixels.
        tip_length: Arrow tip length as fraction of arrow length.
        anti_alias: If True, use anti-aliased drawing.

    Returns:
        Image with arrow drawn.

    Example:
        >>> result = arrow(image, (10, 10), (100, 100), (255, 0, 0))
    """
    result = image.copy()

    # Convert color to BGR
    if len(color) == 4:
        bgr_color = (color[2], color[1], color[0])
    else:
        bgr_color = (color[2], color[1], color[0])

    line_type = cv2.LINE_AA if anti_alias else cv2.LINE_8

    cv2.arrowedLine(
        result,
        start,
        end,
        bgr_color,
        thickness,
        line_type,
        tipLength=tip_length,
    )

    return result
