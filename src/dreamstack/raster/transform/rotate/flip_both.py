"""Flip both directions operation."""

# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def flip_both(
    image: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    """
    Flip image both horizontally and vertically.

    Equivalent to 180 degree rotation.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input image.

    Returns
    -------
    NDArray[np.uint8]
        Flipped image.
    """
    return cv2.flip(image, -1)  # type: ignore[return-value]
