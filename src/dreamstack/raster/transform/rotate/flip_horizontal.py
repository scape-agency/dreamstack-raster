"""Flip horizontal operation."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def flip_horizontal(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Flip image horizontally (mirror).

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input image.

    Returns
    -------
    NDArray[np.uint8]
        Horizontally flipped image.

    Examples
    --------
    >>> mirrored = flip_horizontal(img)
    """
    return cv2.flip(image, 1)
