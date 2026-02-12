"""
Histogram Function
==================

Compute histogram of an image.

"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def histogram(
    image: NDArray[np.uint8],
    bins: int = 256,
    mask: NDArray[np.uint8] | None = None,
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
