# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Histogram Statistics Function
=============================

Compute histogram statistics.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def histogram_stats(
    image: NDArray[np.uint8],
    mask: NDArray[np.uint8] | None = None,
) -> dict[str, float]:
    """Compute histogram statistics.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input image.
    mask : NDArray[np.uint8], optional
        Optional mask.

    Returns
    -------
    dict
        Statistics including mean, std, min, max, median.
    """
    if mask is not None:
        pixels = image[mask > 0]
    else:
        pixels = image.flatten()

    return {
        "mean": float(np.mean(pixels)),
        "std": float(np.std(pixels)),
        "min": float(np.min(pixels)),
        "max": float(np.max(pixels)),
        "median": float(np.median(pixels)),
    }
