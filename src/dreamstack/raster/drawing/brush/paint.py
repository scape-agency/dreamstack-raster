# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Paint
=====

Paint tool for continuous brush application.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel
    from numpy.typing import NDArray

from dreamstack.raster.drawing.brush.brush import Brush
from dreamstack.raster.drawing.brush.stroke import stroke


def paint(
    image: NDArray[np.uint8],
    x: int,
    y: int,
    brush: Brush | None = None,
    *,
    color: tuple[int, int, int, int] | None = None,
) -> NDArray[np.uint8]:
    """Apply a single paint dab at position.

    Simpler interface for single-point painting.

    Args:
        image: Image to paint on.
        x: X coordinate.
        y: Y coordinate.
        brush: Brush configuration.
        color: Optional color override (RGBA).

    Returns:
        Image with paint applied.

    Example:
        >>> result = paint(image, 100, 100, brush, color=(255, 0, 0, 255))
    """
    return stroke(image, [(x, y)], brush, color=color)
