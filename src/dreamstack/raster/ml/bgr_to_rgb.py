"""BGR to RGB conversion."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def bgr_to_rgb(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Convert BGR to RGB color space.

    Parameters
    ----------
    image : NDArray[np.uint8]
        BGR image (OpenCV default).

    Returns
    -------
    NDArray[np.uint8]
        RGB image.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # type: ignore[return-value]
