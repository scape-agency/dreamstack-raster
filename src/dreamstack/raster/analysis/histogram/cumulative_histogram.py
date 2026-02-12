"""
Cumulative Histogram Function
=============================

Compute cumulative histogram.

"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .histogram import histogram


def cumulative_histogram(
    image: NDArray[np.uint8],
    bins: int = 256,
    mask: NDArray[np.uint8] | None = None,
) -> NDArray[np.float64]:
    """Compute cumulative histogram.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input image.
    bins : int, optional
        Number of bins. Default is 256.
    mask : NDArray[np.uint8], optional
        Optional mask.

    Returns
    -------
    NDArray[np.float64]
        Cumulative histogram.
    """
    hist = histogram(image, bins, mask)
    return np.cumsum(hist)
