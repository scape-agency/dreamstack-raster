"""
Extract With Alpha
==================

Function for extracting object with transparent background.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from dreamstack.raster.analysis.contour.info import ContourInfo
from dreamstack.raster.extraction.extract_region import extract_region


def extract_with_alpha(
    image: NDArray[np.uint8],
    contour: ContourInfo,
    margin: int = 25,
    feather: int = 0,
) -> NDArray[np.uint8] | None:
    """Extract object with transparent background (alpha channel).

    Parameters
    ----------
    image : NDArray[np.uint8]
        Source image in BGR format.
    contour : ContourInfo
        Contour information for the object.
    margin : int, optional
        Margin around object. Default 25.
    feather : int, optional
        Feathering pixels for smooth edges. Default 0.

    Returns
    -------
    NDArray[np.uint8] | None
        BGRA image with transparent background, or None if extraction fails.

    Examples
    --------
    >>> obj_rgba = extract_with_alpha(image, contour, feather=2)
    >>> cv2.imwrite("object.png", obj_rgba)  # PNG supports alpha
    """
    # Get bounding region
    x, y, w, h = contour.bounding_rect
    cutout = extract_region(image, x, y, w, h, margin=margin)

    if cutout.size == 0:
        return None

    # Create local coordinates for contour
    offset_x = max(0, x - margin)
    offset_y = max(0, y - margin)
    local_contour = contour.contour.copy()
    local_contour[:, :, 0] -= offset_x
    local_contour[:, :, 1] -= offset_y

    # Create mask
    mask = np.zeros(cutout.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [local_contour], -1, 255, -1)

    # Apply feathering if requested
    if feather > 0:
        mask = cv2.GaussianBlur(mask, (feather * 2 + 1, feather * 2 + 1), 0)

    # Create BGRA image
    bgra = cv2.cvtColor(cutout, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = mask

    return bgra
