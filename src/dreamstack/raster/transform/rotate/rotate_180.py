"""Rotate 180 degrees operation."""

# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def rotate_180(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Rotate image 180 degrees.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input image.

    Returns
    -------
    NDArray[np.uint8]
        Rotated image.
    """
    return cv2.rotate(image, cv2.ROTATE_180)  # type: ignore[return-value]
