# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Color Count Function
====================

Count unique colors in image.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def color_count(
    image: NDArray[np.uint8],
    mask: NDArray[np.uint8] | None = None,
) -> int:
    """Count unique colors in image.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input image.
    mask : NDArray[np.uint8], optional
        Optional mask.

    Returns
    -------
    int
        Number of unique colors.
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

    if image.ndim == 3:
        # Convert to single value per pixel
        unique = np.unique(pixels.astype(np.int64).dot([1, 256, 65536]))
        return len(unique)
    else:
        return len(np.unique(pixels))
