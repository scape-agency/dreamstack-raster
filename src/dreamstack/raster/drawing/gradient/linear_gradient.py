"""
Linear Gradient
===============

Create linear gradient images.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from dreamstack.raster.drawing.gradient.gradient_stop import GradientStop


def linear_gradient(
    width: int,
    height: int,
    start: tuple[int, int],
    end: tuple[int, int],
    stops: list[GradientStop] | None = None,
    *,
    start_color: tuple[int, int, int, int] = (0, 0, 0, 255),
    end_color: tuple[int, int, int, int] = (255, 255, 255, 255),
) -> NDArray[np.uint8]:
    """Create a linear gradient image.

    Args:
        width: Image width.
        height: Image height.
        start: Gradient start point (x, y).
        end: Gradient end point (x, y).
        stops: Optional list of color stops. If provided, overrides start/end colors.
        start_color: Starting color (RGBA) when stops not provided.
        end_color: Ending color (RGBA) when stops not provided.

    Returns:
        RGBA gradient image.

    Example:
        >>> # Simple black to white horizontal
        >>> grad = linear_gradient(512, 512, (0, 256), (512, 256))
        >>>
        >>> # Multi-stop gradient
        >>> stops = [
        ...     GradientStop(0.0, (255, 0, 0, 255)),
        ...     GradientStop(0.5, (0, 255, 0, 255)),
        ...     GradientStop(1.0, (0, 0, 255, 255)),
        ... ]
        >>> rainbow = linear_gradient(512, 512, (0, 0), (512, 0), stops)
    """
    # Default stops
    if stops is None:
        stops = [
            GradientStop(0.0, start_color),
            GradientStop(1.0, end_color),
        ]

    # Sort stops by position
    stops = sorted(stops, key=lambda s: s.position)

    # Create coordinate grids
    y_coords, x_coords = np.mgrid[0:height, 0:width]

    # Calculate gradient direction
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = np.sqrt(dx**2 + dy**2)

    if length < 1e-6:
        # Fallback to horizontal
        t = x_coords.astype(np.float32) / max(1, width - 1)
    else:
        # Project coordinates onto gradient line
        px = x_coords - start[0]
        py = y_coords - start[1]
        t = (px * dx + py * dy) / (length**2)
        t = np.clip(t, 0, 1)

    # Interpolate colors
    image = _interpolate_colors(t, stops)

    return image


def _interpolate_colors(
    t: NDArray[np.float32],
    stops: list[GradientStop],
) -> NDArray[np.uint8]:
    """Interpolate colors based on t values and stops."""
    h, w = t.shape
    result = np.zeros((h, w, 4), dtype=np.float32)

    for i in range(len(stops) - 1):
        s1 = stops[i]
        s2 = stops[i + 1]

        # Mask for this segment
        mask = (t >= s1.position) & (t <= s2.position)

        if not np.any(mask):
            continue

        # Local interpolation factor
        segment_t = (t[mask] - s1.position) / max(0.001, s2.position - s1.position)

        # Interpolate each channel
        for c in range(4):
            result[mask, c] = s1.color[c] * (1 - segment_t) + s2.color[c] * segment_t

    # Handle edges
    result[t <= stops[0].position] = stops[0].color
    result[t >= stops[-1].position] = stops[-1].color

    return np.clip(result, 0, 255).astype(np.uint8)
