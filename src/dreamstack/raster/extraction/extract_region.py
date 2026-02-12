"""
Extract Region
==============

Function for extracting rectangular regions from images.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def extract_region(
    image: NDArray[np.uint8],
    x: int,
    y: int,
    width: int,
    height: int,
    margin: int = 0,
) -> NDArray[np.uint8]:
    """Extract a rectangular region from an image.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Source image.
    x, y : int
        Top-left corner coordinates.
    width, height : int
        Region dimensions.
    margin : int, optional
        Margin to add around region. Default 0.

    Returns
    -------
    NDArray[np.uint8]
        Extracted region.

    Examples
    --------
    >>> region = extract_region(image, 100, 100, 200, 200, margin=10)
    """
    h, w = image.shape[:2]

    # Apply margin and clamp to image bounds
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(w, x + width + margin)
    y2 = min(h, y + height + margin)

    return image[y1:y2, x1:x2].copy()
