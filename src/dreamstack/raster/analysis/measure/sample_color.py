# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Sample Color Function
=====================

Sample color at a specific pixel.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def sample_color(
    image: NDArray[np.uint8],
    x: int,
    y: int,
) -> tuple[int, int, int]:
    """Sample color at a specific pixel.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input image (BGR).
    x : int
        X coordinate.
    y : int
        Y coordinate.

    Returns
    -------
    tuple
        BGR color values.
    """
    if x < 0 or x >= image.shape[1] or y < 0 or y >= image.shape[0]:
        raise ValueError(f"Coordinates ({x}, {y}) out of bounds")

    if image.ndim == 2:
        val = int(image[y, x])
        return (val, val, val)

    px = image[y, x]
    return (int(px[0]), int(px[1]), int(px[2]))
