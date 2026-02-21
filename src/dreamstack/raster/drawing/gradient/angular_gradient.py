# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Angular Gradient
================

Create angular (conic) gradient images.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

# =============================================================================
# Type Checking Imports
# =============================================================================

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel
    from numpy.typing import NDArray

from dreamstack.raster.drawing.gradient.gradient_stop import GradientStop


def angular_gradient(
    width: int,
    height: int,
    center: tuple[int, int],
    stops: list[GradientStop] | None = None,
    *,
    start_angle: float = 0.0,
    start_color: tuple[int, int, int, int] = (0, 0, 0, 255),
    end_color: tuple[int, int, int, int] = (255, 255, 255, 255),
) -> NDArray[np.uint8]:
    """Create an angular (conic) gradient image.

    Colors sweep around the center point.

    Args:
        width: Image width.
        height: Image height.
        center: Gradient center point (x, y).
        stops: Optional list of color stops.
        start_angle: Starting angle in degrees (0 = right, counter-clockwise).
        start_color: Starting color when stops not provided.
        end_color: Ending color when stops not provided.

    Returns:
        RGBA gradient image.

    Example:
        >>> grad = angular_gradient(512, 512, (256, 256))
    """
    # Default stops
    if stops is None:
        stops = [
            GradientStop(0.0, start_color),
            GradientStop(1.0, end_color),
        ]

    stops = sorted(stops, key=lambda s: s.position)

    # Create coordinate grids
    y_coords, x_coords = np.mgrid[0:height, 0:width]

    # Calculate angle from center
    dx = x_coords - center[0]
    dy = y_coords - center[1]

    # atan2 returns radians, convert to [0, 1]
    angles = np.arctan2(dy, dx)
    start_rad = np.radians(start_angle)
    angles = (angles - start_rad) % (2 * np.pi)
    t = (angles / (2 * np.pi)).astype(np.float32)

    # Interpolate colors
    return _interpolate_angular(t, stops)


def _interpolate_angular(
    t: NDArray[np.float32],
    stops: list[GradientStop],
) -> NDArray[np.uint8]:
    """Interpolate colors for angular gradient."""
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
