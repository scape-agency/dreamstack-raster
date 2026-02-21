# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Create Gradient
===============

Internal helper to generate gradient backgrounds.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from dreamstack.raster.effects.background.gradient_config import (
    GradientDirection,
)

# =============================================================================
# Type Checking Imports
# =============================================================================

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel
    from numpy.typing import NDArray


def _create_gradient(
    width: int,
    height: int,
    start_color: tuple[int, int, int],
    end_color: tuple[int, int, int],
    direction: GradientDirection,
    center: tuple[float, float] = (0.5, 0.5),
) -> NDArray[np.uint8]:
    """Create a gradient image.

    Internal function to generate gradient backgrounds.
    """
    # Create coordinate grids
    x = np.linspace(0, 1, width)  # type: ignore[call-arg]
    y = np.linspace(0, 1, height)  # type: ignore[call-arg]
    xx, yy = np.meshgrid(x, y)

    # Calculate interpolation factor based on direction
    if direction == "horizontal":
        t = xx
    elif direction == "vertical":
        t = yy
    elif direction == "diagonal":
        t = (xx + yy) / 2
    elif direction == "radial":
        cx, cy = center
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        t = np.clip(dist / np.sqrt(2), 0, 1)
    else:
        t = yy  # Default to vertical

    # Expand dimensions for broadcasting
    t = np.expand_dims(t, axis=-1)

    # Interpolate colors
    start = np.array(start_color, dtype=np.float32)
    end = np.array(end_color, dtype=np.float32)

    gradient = start + t * (end - start)

    return gradient.astype(np.uint8)
