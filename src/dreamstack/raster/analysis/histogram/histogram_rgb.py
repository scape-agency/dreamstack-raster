"""
RGB Histogram Function
======================

Compute separate histograms for R, G, B channels.

"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def histogram_rgb(
    image: NDArray[np.uint8],
    bins: int = 256,
    mask: NDArray[np.uint8] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute separate histograms for R, G, B channels.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input BGR image.
    bins : int, optional
        Number of bins. Default is 256.
    mask : NDArray[np.uint8], optional
        Optional mask.

    Returns
    -------
    tuple
        (blue_hist, green_hist, red_hist)
    """
    b_hist = (
        cv2.calcHist([image], [0], mask, [bins], [0, 256])
        .flatten()
        .astype(np.float64)
    )
    g_hist = (
        cv2.calcHist([image], [1], mask, [bins], [0, 256])
        .flatten()
        .astype(np.float64)
    )
    r_hist = (
        cv2.calcHist([image], [2], mask, [bins], [0, 256])
        .flatten()
        .astype(np.float64)
    )
    return (b_hist, g_hist, r_hist)
