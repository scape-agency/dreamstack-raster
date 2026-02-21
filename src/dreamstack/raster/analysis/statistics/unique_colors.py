# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Unique Colors Function
======================

Get list of unique colors.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def unique_colors(
    image: NDArray[np.uint8],
    max_colors: int = 1000,
    mask: NDArray[np.uint8] | None = None,
) -> list[tuple[int, ...]]:
    """Get list of unique colors.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input image.
    max_colors : int, optional
        Maximum colors to return. Default is 1000.
    mask : NDArray[np.uint8], optional
        Optional mask.

    Returns
    -------
    list
        List of unique colors as tuples.
    """
    if mask is not None:
        if image.ndim == 3:
            pixels = image[mask > 0]
        else:
            pixels = image[mask > 0]
    else:
        if image.ndim == 3:
            pixels = image.reshape(-1, image.shape[2])
        else:
            pixels = image.flatten()

    unique = np.unique(pixels, axis=0)

    if len(unique) > max_colors:
        unique = unique[:max_colors]

    if image.ndim == 3:
        return [tuple(int(v) for v in c) for c in unique]
    else:
        return [(int(c),) for c in unique]
