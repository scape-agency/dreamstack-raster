# -*- coding: utf-8 -*-
# pyright: reportArgumentType=false


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Apply Background Mask
=====================

Function for replacing background with a solid color based on masking.
"""

# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from dreamstack.raster.analysis.contour.operations import find_largest_contour
from dreamstack.raster.analysis.preprocessing.operations import detect_edges


def apply_background_mask(
    image: NDArray[np.uint8],
    background_color: tuple[int, int, int],
    threshold_offset: int = 20,
) -> NDArray[np.uint8]:
    """Replace background with a solid color based on masking.

    Creates a mask of the main object and replaces surrounding
    pixels with the specified background color.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input image in BGR format.
    background_color : tuple[int, int, int]
        Replacement background color (B, G, R).
    threshold_offset : int, optional
        Color tolerance for background detection. Default 20.

    Returns
    -------
    NDArray[np.uint8]
        Image with masked background.

    Examples
    --------
    >>> masked = apply_background_mask(image, (255, 255, 255))  # White bg
    """
    # Detect edges and find main object
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Use threshold_offset to adjust edge detection sensitivity
    base_low = 50
    base_high = 150
    edges = detect_edges(
        gray,
        low_threshold=max(0, base_low - threshold_offset),
        high_threshold=max(base_low, base_high - threshold_offset),
    )

    # Find largest contour
    largest = find_largest_contour(edges)
    if largest is None:
        return image.copy()

    # Create mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [largest.contour], -1, 255, -1)

    # Apply mask
    result = image.copy()
    result[mask == 0] = background_color

    return result
