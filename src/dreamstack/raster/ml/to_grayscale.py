"""To grayscale operation."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def to_grayscale(
    image: NDArray[np.uint8],
    method: str = "luminance",
) -> NDArray[np.uint8]:
    """Convert image to grayscale.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input BGR/RGB image.
    method : str, optional
        Conversion method:
        - "luminance": Perceptual weights (default)
        - "average": Simple average
        - "cv2": OpenCV default

    Returns
    -------
    NDArray[np.uint8]
        Grayscale image.
    """
    if image.ndim == 2:
        return image

    if method == "cv2":
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    elif method == "average":
        return np.mean(image, axis=2).astype(np.uint8)

    else:  # luminance
        # Assumes BGR
        b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)
