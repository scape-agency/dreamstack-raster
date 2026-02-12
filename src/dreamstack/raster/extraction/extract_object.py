"""
Extract Object
==============

Function for extracting a single object based on its contour.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from dreamstack.raster.analysis.contour.info import ContourInfo
from dreamstack.raster.extraction.extract_region import extract_region


def extract_object(
    image: NDArray[np.uint8],
    contour: ContourInfo,
    margin: int = 25,
    min_dimension: int = 24,
) -> NDArray[np.uint8] | None:
    """Extract a single object based on its contour.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Source image.
    contour : ContourInfo
        Contour information for the object.
    margin : int, optional
        Margin to add around object. Default 25.
    min_dimension : int, optional
        Minimum width/height. Returns None if smaller. Default 24.

    Returns
    -------
    NDArray[np.uint8] | None
        Extracted object image, or None if too small or empty.

    Examples
    --------
    >>> for contour in contours:
    ...     obj = extract_object(image, contour, margin=30)
    ...     if obj is not None:
    ...         process(obj)
    """
    x, y, w, h = contour.bounding_rect

    # Extract with margin
    cutout = extract_region(image, x, y, w, h, margin=margin)

    if cutout.size == 0:
        return None

    # Validate dimensions
    cutout_h, cutout_w = cutout.shape[:2]
    if cutout_w < min_dimension or cutout_h < min_dimension:
        return None

    return cutout
