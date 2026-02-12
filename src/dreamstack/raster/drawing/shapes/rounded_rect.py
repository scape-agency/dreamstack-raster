"""
Rounded Rectangle
=================

Draw rounded rectangles on images.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def rounded_rect(
    image: NDArray[np.uint8],
    x: int,
    y: int,
    width: int,
    height: int,
    radius: int,
    color: tuple[int, int, int] | tuple[int, int, int, int] = (0, 0, 0, 255),
    *,
    thickness: int = 1,
    filled: bool = False,
) -> NDArray[np.uint8]:
    """Draw a rounded rectangle on an image.

    Args:
        image: Image to draw on.
        x: Top-left x coordinate.
        y: Top-left y coordinate.
        width: Rectangle width.
        height: Rectangle height.
        radius: Corner radius in pixels.
        color: Rectangle color (RGB or RGBA).
        thickness: Border thickness (ignored if filled).
        filled: If True, fill the rectangle.

    Returns:
        Image with rounded rectangle drawn.

    Example:
        >>> result = rounded_rect(image, 10, 10, 100, 50, 10, (255, 0, 0))
    """
    result = image.copy()

    # Clamp radius
    radius = min(radius, width // 2, height // 2)

    # Convert color to BGR
    if len(color) == 4:
        bgr_color = (color[2], color[1], color[0])
    else:
        bgr_color = (color[2], color[1], color[0])

    x1, y1 = x, y
    x2, y2 = x + width, y + height

    thick = -1 if filled else thickness

    # Draw rectangles (body)
    cv2.rectangle(result, (x1 + radius, y1), (x2 - radius, y2), bgr_color, thick)
    cv2.rectangle(result, (x1, y1 + radius), (x2, y2 - radius), bgr_color, thick)

    # Draw corner circles
    cv2.circle(
        result,
        (x1 + radius, y1 + radius),
        radius,
        bgr_color,
        thick,
        cv2.LINE_AA,
    )
    cv2.circle(
        result,
        (x2 - radius, y1 + radius),
        radius,
        bgr_color,
        thick,
        cv2.LINE_AA,
    )
    cv2.circle(
        result,
        (x1 + radius, y2 - radius),
        radius,
        bgr_color,
        thick,
        cv2.LINE_AA,
    )
    cv2.circle(
        result,
        (x2 - radius, y2 - radius),
        radius,
        bgr_color,
        thick,
        cv2.LINE_AA,
    )

    return result
