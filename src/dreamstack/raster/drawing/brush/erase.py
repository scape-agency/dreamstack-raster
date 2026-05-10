# -*- coding: utf-8 -*-
# pyright: reportArgumentType=false, reportReturnType=false


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Erase
=====

Eraser tool for removing pixels.

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
    # pylint: disable=import-outside-toplevel
    from numpy.typing import NDArray

from dreamstack.raster.drawing.brush.brush import Brush


def erase(
    image: NDArray[np.uint8],
    points: list[tuple[int, int]],
    *,
    size: int = 20,
    hardness: float = 0.5,
    opacity: float = 1.0,
) -> NDArray[np.uint8]:
    """Erase pixels along a path.

    Sets alpha to transparent along the stroke path.

    Args:
        image: Image to erase from.
        points: List of (x, y) coordinates.
        size: Eraser size in pixels.
        hardness: Edge hardness (0.0 = soft, 1.0 = hard).
        opacity: Erase strength (1.0 = fully transparent).

    Returns:
        Image with erased area.

    Example:
        >>> result = erase(image, [(10, 10), (100, 100)], size=30)
    """
    # Ensure BGRA
    if image.ndim == 2:
        result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    elif image.shape[2] == 3:
        result = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    else:
        result = image.copy()

    if len(points) < 1:
        return result

    # Create eraser brush
    brush = Brush(
        size=size,
        hardness=hardness,
        opacity=opacity,
        flow=1.0,
        spacing=0.1,
    )

    tip = brush.create_tip()
    tip_h, tip_w = tip.shape

    h, w = result.shape[:2]

    # Interpolate points
    all_points = _interpolate_erase_points(points, size * 0.1)

    # Apply eraser
    for px, py in all_points:
        px, py = int(px), int(py)

        half_h, half_w = tip_h // 2, tip_w // 2

        y1 = max(0, py - half_h)
        y2 = min(h, py + half_h + 1)
        x1 = max(0, px - half_w)
        x2 = min(w, px + half_w + 1)

        ty1 = half_h - (py - y1)
        ty2 = ty1 + (y2 - y1)
        tx1 = half_w - (px - x1)
        tx2 = tx1 + (x2 - x1)

        if y2 <= y1 or x2 <= x1:
            continue

        tip_region = tip[ty1:ty2, tx1:tx2]
        erase_amount = tip_region * opacity * 255

        # Reduce alpha
        current_alpha = result[y1:y2, x1:x2, 3].astype(np.float32)
        new_alpha = current_alpha - erase_amount
        result[y1:y2, x1:x2, 3] = np.clip(new_alpha, 0, 255).astype(np.uint8)

    return result


def _interpolate_erase_points(
    points: list[tuple[int, int]],
    spacing: float,
) -> list[tuple[float, float]]:
    """Interpolate points for smooth erasing."""
    if len(points) < 2:
        return [(float(p[0]), float(p[1])) for p in points]

    result = []
    accumulated = 0.0

    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]

        dx = x2 - x1
        dy = y2 - y1
        segment_length = np.sqrt(dx**2 + dy**2)

        if segment_length < 1e-6:
            continue

        if i == 0:
            result.append((float(x1), float(y1)))

        while accumulated + spacing <= segment_length:
            accumulated += spacing
            t = accumulated / segment_length
            px = x1 + dx * t
            py = y1 + dy * t
            result.append((px, py))

        accumulated -= segment_length

    return result
