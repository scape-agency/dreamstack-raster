# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Color Sampler Function
======================

Sample average color in a region.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .sample_color import sample_color


def color_sampler(
    image: NDArray[np.uint8],
    x: int,
    y: int,
    radius: int = 0,
) -> tuple[int, int, int]:
    """Sample average color in a region.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input image.
    x : int
        Center X coordinate.
    y : int
        Center Y coordinate.
    radius : int, optional
        Sample radius. Default is 0 (single pixel).

    Returns
    -------
    tuple
        Average BGR color.
    """
    if radius == 0:
        return sample_color(image, x, y)

    h, w = image.shape[:2]
    x1 = max(0, x - radius)
    y1 = max(0, y - radius)
    x2 = min(w, x + radius + 1)
    y2 = min(h, y + radius + 1)

    region = image[y1:y2, x1:x2]
    avg = np.mean(region, axis=(0, 1))

    if image.ndim == 2:
        val = int(avg)
        return (val, val, val)

    return (int(avg[0]), int(avg[1]), int(avg[2]))
