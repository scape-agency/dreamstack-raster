"""Flip vertical operation."""

# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def flip_vertical(
    image: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    """Flip image vertically.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input image.

    Returns
    -------
    NDArray[np.uint8]
        Vertically flipped image.
    """
    return cv2.flip(image, 0)  # type: ignore[return-value]
