"""Rotate 270 degrees operation."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def rotate_270(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Rotate image 270 degrees counter-clockwise (90 clockwise).

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input image.

    Returns
    -------
    NDArray[np.uint8]
        Rotated image with swapped dimensions.
    """
    return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
