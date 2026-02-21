# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Stroke
======

Draw brush strokes along paths.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from dreamstack.raster.drawing.brush.brush import Brush

# =============================================================================
# Type Checking Imports
# =============================================================================

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel
    from numpy.typing import NDArray


def stroke(
    image: NDArray[np.uint8],
    points: list[tuple[int, int]],
    brush: Brush | None = None,
    *,
    color: tuple[int, int, int, int] | None = None,
) -> NDArray[np.uint8]:
    """Draw a brush stroke along a path.

    Applies brush dabs along the given points to create a stroke.

    Args:
        image: Image to draw on (modified in place).
        points: List of (x, y) coordinates defining the stroke path.
        brush: Brush configuration to use.
        color: Optional color override (RGBA).

    Returns:
        Image with stroke applied.

    Example:
        >>> brush = Brush(size=10, hardness=0.5)
        >>> result = stroke(image, [(10, 10), (100, 100), (200, 50)], brush)
    """
    if brush is None:
        brush = Brush()

    if color is not None:
        brush.color = color

    # Ensure BGRA
    import cv2  # pylint: disable=import-outside-toplevel

    if image.ndim == 2:
        result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    elif image.shape[2] == 3:
        result = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    else:
        result = image.copy()

    if len(points) < 1:
        return result

    # Create brush tip
    tip = brush.create_tip()
    _tip_h, _tip_w = tip.shape  # Unused but kept for reference

    # Interpolate points based on spacing
    all_points = _interpolate_points(points, brush.size * brush.spacing)

    # Apply each dab
    for px, py in all_points:
        _apply_dab(result, tip, int(px), int(py), brush)

    return result


def _interpolate_points(
    points: list[tuple[int, int]],
    spacing: float,
) -> list[tuple[float, float]]:
    """Interpolate points along path with given spacing."""
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

        # Add start point of segment
        if i == 0:
            result.append((float(x1), float(y1)))

        # Add interpolated points
        while accumulated + spacing <= segment_length:
            accumulated += spacing
            t = accumulated / segment_length
            px = x1 + dx * t
            py = y1 + dy * t
            result.append((px, py))

        accumulated -= segment_length

    return result


def _apply_dab(
    image: NDArray[np.uint8],
    tip: NDArray[np.float32],
    x: int,
    y: int,
    brush: Brush,
) -> None:
    """Apply a single brush dab at position."""
    h, w = image.shape[:2]
    tip_h, tip_w = tip.shape

    # Calculate bounds
    half_h, half_w = tip_h // 2, tip_w // 2

    # Image region
    y1 = max(0, y - half_h)
    y2 = min(h, y + half_h + 1)
    x1 = max(0, x - half_w)
    x2 = min(w, x + half_w + 1)

    # Tip region
    ty1 = half_h - (y - y1)
    ty2 = ty1 + (y2 - y1)
    tx1 = half_w - (x - x1)
    tx2 = tx1 + (x2 - x1)

    if y2 <= y1 or x2 <= x1:
        return

    # Get regions
    img_region = image[y1:y2, x1:x2].astype(np.float32)
    tip_region = tip[ty1:ty2, tx1:tx2]

    # Apply color with opacity
    opacity = brush.opacity * tip_region[:, :, np.newaxis]

    # Convert color to BGR order (OpenCV)
    brush_color = np.array(
        [
            brush.color[2],  # B
            brush.color[1],  # G
            brush.color[0],  # R
            brush.color[3],  # A
        ],
        dtype=np.float32,
    )

    # Blend
    blended = img_region * (1 - opacity) + brush_color * opacity
    image[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
