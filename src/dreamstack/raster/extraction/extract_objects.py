"""
Extract Objects
===============

Function for extracting all objects from an image.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from dreamstack.raster.analysis.contour.info import ContourInfo
from dreamstack.raster.analysis.contour.operations import (
    analyze_contours,
    filter_by_area,
    find_contours,
)
from dreamstack.raster.analysis.preprocessing.operations import (
    preprocess_for_contours,
)
from dreamstack.raster.extraction.extract_object import extract_object


def extract_objects(
    image: NDArray[np.uint8],
    binary_mask: NDArray[np.uint8] | None = None,
    min_area_ratio: float = 0.0002,
    max_area_ratio: float = 0.95,
    margin: int = 25,
    min_dimension: int = 24,
) -> list[tuple[NDArray[np.uint8], ContourInfo]]:
    """Extract all objects from an image.

    Performs preprocessing (if no mask provided), finds contours,
    filters by area, and extracts each object.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Source image in BGR format.
    binary_mask : NDArray[np.uint8] | None, optional
        Pre-computed binary mask. If None, creates one using
        default preprocessing.
    min_area_ratio : float, optional
        Minimum object area as ratio of image area. Default 0.0002.
    max_area_ratio : float, optional
        Maximum object area as ratio of image area. Default 0.95.
    margin : int, optional
        Margin around extracted objects. Default 25.
    min_dimension : int, optional
        Minimum object dimension. Default 24.

    Returns
    -------
    list[tuple[NDArray[np.uint8], ContourInfo]]
        List of (extracted_image, contour_info) tuples,
        sorted by area (largest first).

    Examples
    --------
    >>> extractions = extract_objects(image)
    >>> for obj_image, contour in extractions:
    ...     cv2.imwrite(f"object_{contour.area:.0f}.png", obj_image)
    """
    # Get or create binary mask
    if binary_mask is None:
        results = preprocess_for_contours(image)
        binary_mask = results["threshold"]

    # Find and filter contours
    h, w = image.shape[:2]
    image_area = h * w

    raw_contours = find_contours(binary_mask)
    analyzed = analyze_contours(raw_contours)
    filtered = filter_by_area(
        analyzed,
        image_area,
        min_ratio=min_area_ratio,
        max_ratio=max_area_ratio,
    )

    # Extract each object
    extractions = []
    for contour in filtered:
        obj = extract_object(image, contour, margin=margin, min_dimension=min_dimension)
        if obj is not None:
            extractions.append((obj, contour))

    return extractions
