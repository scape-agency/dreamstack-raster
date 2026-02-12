# -*- coding: utf-8 -*-

"""
Histogram Analysis Module
=========================

Histogram computation and analysis functions.

"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray


def histogram(
    image: NDArray[np.uint8],
    bins: int = 256,
    mask: Optional[NDArray[np.uint8]] = None,
) -> NDArray[np.float64]:
    """Compute histogram of an image.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input image.
    bins : int, optional
        Number of histogram bins. Default is 256.
    mask : NDArray[np.uint8], optional
        Optional mask.

    Returns
    -------
    NDArray[np.float64]
        Histogram values.
    """
    if image.ndim == 3:
        # Convert to grayscale for single histogram
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    hist = cv2.calcHist([gray], [0], mask, [bins], [0, 256])
    return hist.flatten()


def histogram_rgb(
    image: NDArray[np.uint8],
    bins: int = 256,
    mask: Optional[NDArray[np.uint8]] = None,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
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
    b_hist = cv2.calcHist([image], [0], mask, [bins], [0, 256]).flatten()
    g_hist = cv2.calcHist([image], [1], mask, [bins], [0, 256]).flatten()
    r_hist = cv2.calcHist([image], [2], mask, [bins], [0, 256]).flatten()
    return (b_hist, g_hist, r_hist)


def histogram_luminosity(
    image: NDArray[np.uint8],
    bins: int = 256,
    mask: Optional[NDArray[np.uint8]] = None,
) -> NDArray[np.float64]:
    """Compute luminosity histogram.

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
        Luminosity histogram.
    """
    if image.ndim == 3:
        # Convert to luminosity
        b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        lum = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)
    else:
        lum = image

    return histogram(lum, bins, mask)


def cumulative_histogram(
    image: NDArray[np.uint8],
    bins: int = 256,
    mask: Optional[NDArray[np.uint8]] = None,
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


def histogram_stats(
    image: NDArray[np.uint8],
    mask: Optional[NDArray[np.uint8]] = None,
) -> Dict[str, float]:
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
