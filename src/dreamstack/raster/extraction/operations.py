# -*- coding: utf-8 -*-

"""
Extraction Operations
=====================

Functional API for object extraction operations.
Provides stateless functions for extracting objects from images.
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

from dreamstack.raster.analysis.contour.info import ContourInfo
from dreamstack.raster.analysis.contour.operations import (
    analyze_contours,
    filter_by_area,
    find_contours,
    find_largest_contour,
)
from dreamstack.raster.analysis.preprocessing.operations import (
    detect_edges,
    preprocess_for_contours,
)


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


def extract_objects(
    image: NDArray[np.uint8],
    binary_mask: NDArray[np.uint8] | None = None,
    min_area_ratio: float = 0.0002,
    max_area_ratio: float = 0.95,
    margin: int = 25,
    min_dimension: int = 24,
) -> List[Tuple[NDArray[np.uint8], ContourInfo]]:
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
        analyzed, image_area, min_ratio=min_area_ratio, max_ratio=max_area_ratio
    )

    # Extract each object
    extractions = []
    for contour in filtered:
        obj = extract_object(image, contour, margin=margin, min_dimension=min_dimension)
        if obj is not None:
            extractions.append((obj, contour))

    return extractions


def apply_background_mask(
    image: NDArray[np.uint8],
    background_color: Tuple[int, int, int],
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
    edges = detect_edges(gray)

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
