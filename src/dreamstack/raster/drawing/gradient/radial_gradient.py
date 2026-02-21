# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Radial Gradient
===============

Create radial gradient images.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from dreamstack.raster.drawing.gradient.gradient_stop import GradientStop


def radial_gradient(
    width: int,
    height: int,
    center: tuple[int, int],
    radius: float,
    stops: list[GradientStop] | None = None,
    *,
    center_color: tuple[int, int, int, int] = (255, 255, 255, 255),
    edge_color: tuple[int, int, int, int] = (0, 0, 0, 255),
    aspect_ratio: float = 1.0,
) -> NDArray[np.uint8]:
    """Create a radial gradient image.

    Args:
        width: Image width.
        height: Image height.
        center: Gradient center point (x, y).
        radius: Gradient radius in pixels.
        stops: Optional list of color stops.
        center_color: Center color when stops not provided.
        edge_color: Edge color when stops not provided.
        aspect_ratio: Aspect ratio (>1 = wider, <1 = taller).

    Returns:
        RGBA gradient image.

    Example:
        >>> grad = radial_gradient(512, 512, (256, 256), 200)
    """
    # Default stops
    if stops is None:
        stops = [
            GradientStop(0.0, center_color),
            GradientStop(1.0, edge_color),
        ]

    stops = sorted(stops, key=lambda s: s.position)

    # Create coordinate grids
    y_coords, x_coords = np.mgrid[0:height, 0:width]

    # Calculate distance from center
    dx = (x_coords - center[0]) / aspect_ratio
    dy = y_coords - center[1]
    distance = np.sqrt(dx**2 + dy**2)

    # Normalize to [0, 1]
    t = np.clip(distance / max(1, radius), 0, 1).astype(np.float32)

    # Interpolate colors
    return _interpolate_radial(t, stops)


def _interpolate_radial(
    t: NDArray[np.float32],
    stops: list[GradientStop],
) -> NDArray[np.uint8]:
    """Interpolate colors for radial gradient."""
    h, w = t.shape
    result = np.zeros((h, w, 4), dtype=np.float32)

    for i in range(len(stops) - 1):
        s1 = stops[i]
        s2 = stops[i + 1]

        mask = (t >= s1.position) & (t <= s2.position)

        if not np.any(mask):
            continue

        segment_t = (t[mask] - s1.position) / max(
            0.001, s2.position - s1.position
        )

        for c in range(4):
            result[mask, c] = (
                s1.color[c] * (1 - segment_t) + s2.color[c] * segment_t
            )

    result[t <= stops[0].position] = stops[0].color
    result[t >= stops[-1].position] = stops[-1].color

    return np.clip(result, 0, 255).astype(np.uint8)
