"""
Star
====

Draw star shapes on images.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def star(
    image: NDArray[np.uint8],
    center: tuple[int, int],
    outer_radius: int,
    inner_radius: int,
    points: int = 5,
    color: tuple[int, int, int] | tuple[int, int, int, int] = (0, 0, 0, 255),
    *,
    angle: float = -90.0,
    thickness: int = 1,
    filled: bool = False,
    anti_alias: bool = True,
) -> NDArray[np.uint8]:
    """Draw a star shape on an image.

    Args:
        image: Image to draw on.
        center: Star center (x, y).
        outer_radius: Radius to outer points.
        inner_radius: Radius to inner points.
        points: Number of star points.
        color: Star color (RGB or RGBA).
        angle: Starting angle in degrees (-90 puts first point at top).
        thickness: Border thickness (ignored if filled).
        filled: If True, fill the star.
        anti_alias: If True, use anti-aliased drawing.

    Returns:
        Image with star drawn.

    Example:
        >>> # 5-pointed star
        >>> result = star(image, (100, 100), 50, 20, 5, (255, 215, 0), filled=True)
    """
    result = image.copy()
    cx, cy = center

    # Generate star vertices
    vertices = []
    angle_step = 360.0 / points

    for i in range(points):
        # Outer point
        outer_angle = np.radians(angle + i * angle_step)
        ox = cx + int(outer_radius * np.cos(outer_angle))
        oy = cy + int(outer_radius * np.sin(outer_angle))
        vertices.append((ox, oy))

        # Inner point
        inner_angle = np.radians(angle + i * angle_step + angle_step / 2)
        ix = cx + int(inner_radius * np.cos(inner_angle))
        iy = cy + int(inner_radius * np.sin(inner_angle))
        vertices.append((ix, iy))

    # Convert color to BGR
    if len(color) == 4:
        bgr_color = (color[2], color[1], color[0])
    else:
        bgr_color = (color[2], color[1], color[0])

    pts = np.array(vertices, dtype=np.int32).reshape((-1, 1, 2))
    line_type = cv2.LINE_AA if anti_alias else cv2.LINE_8

    if filled:
        cv2.fillPoly(result, [pts], bgr_color, line_type)
    else:
        cv2.polylines(result, [pts], True, bgr_color, thickness, line_type)

    return result
