"""Rotate 90 degrees operation."""

# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def rotate_90(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Rotate image 90 degrees counter-clockwise.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input image.

    Returns
    -------
    NDArray[np.uint8]
        Rotated image with swapped dimensions.
    """
    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)  # type: ignore[return-value]
