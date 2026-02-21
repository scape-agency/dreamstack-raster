"""RGB to BGR conversion."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def rgb_to_bgr(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Convert RGB to BGR color space.

    Parameters
    ----------
    image : NDArray[np.uint8]
        RGB image.

    Returns
    -------
    NDArray[np.uint8]
        BGR image (OpenCV default).
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # type: ignore[return-value]
