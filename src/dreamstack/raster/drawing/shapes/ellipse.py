# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Ellipse
=======

Draw ellipses on images.

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
    from numpy.typing import NDArray


def ellipse(
    image: NDArray[np.uint8],
    center: tuple[int, int],
    axes: tuple[int, int],
    color: tuple[int, int, int] | tuple[int, int, int, int] = (0, 0, 0, 255),
    *,
    angle: float = 0.0,
    start_angle: float = 0.0,
    end_angle: float = 360.0,
    thickness: int = 1,
    filled: bool = False,
    anti_alias: bool = True,
) -> NDArray[np.uint8]:
    """Draw an ellipse on an image.

    Args:
        image: Image to draw on.
        center: Center point (x, y).
        axes: Axes lengths (half-width, half-height).
        color: Ellipse color (RGB or RGBA).
        angle: Rotation angle in degrees.
        start_angle: Arc start angle in degrees.
        end_angle: Arc end angle in degrees.
        thickness: Border thickness (ignored if filled).
        filled: If True, fill the ellipse.
        anti_alias: If True, use anti-aliased drawing.

    Returns:
        Image with ellipse drawn.

    Example:
        >>> # Draw circle
        >>> result = ellipse(image, (100, 100), (50, 50), (255, 0, 0))
        >>> # Draw filled ellipse
        >>> result = ellipse(image, (100, 100), (80, 40), (0, 255, 0), filled=True)
    """
    result = image.copy()

    # Convert color to BGR
    if len(color) == 4:
        bgr_color = (color[2], color[1], color[0])
    else:
        bgr_color = (color[2], color[1], color[0])

    thick = -1 if filled else thickness
    line_type = cv2.LINE_AA if anti_alias else cv2.LINE_8

    cv2.ellipse(
        result,
        center,
        axes,
        angle,
        start_angle,
        end_angle,
        bgr_color,
        thick,
        line_type,
    )

    return result
