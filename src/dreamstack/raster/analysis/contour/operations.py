"""
Contour Operations
==================

Functional API for contour detection and analysis operations.
Provides stateless functions for finding, filtering, and analyzing contours.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2  # pylint: disable=no-member
import numpy as np
from numpy.typing import NDArray

from dreamstack.raster.analysis.contour.info import ContourInfo


def find_contours(
    binary_image: NDArray[np.uint8],
    mode: int = cv2.RETR_EXTERNAL,  # pylint: disable=no-member
    method: int = cv2.CHAIN_APPROX_SIMPLE,  # pylint: disable=no-member
) -> list[NDArray[np.int32]]:
    """Find contours in a binary image.

    Uses OpenCV's findContours to detect object boundaries
    in a thresholded or edge-detected image.

    Parameters
    ----------
    binary_image : NDArray[np.uint8]
        Binary (thresholded) image where objects are white (255)
        and background is black (0).
    mode : int, optional
        Contour retrieval mode:
        - cv2.RETR_EXTERNAL: Only outer contours
        - cv2.RETR_LIST: All contours without hierarchy
        - cv2.RETR_TREE: Full hierarchy
        Default is cv2.RETR_EXTERNAL.
    method : int, optional
        Contour approximation method:
        - cv2.CHAIN_APPROX_SIMPLE: Compress segments
        - cv2.CHAIN_APPROX_NONE: Store all points
        Default is cv2.CHAIN_APPROX_SIMPLE.

    Returns
    -------
    list[NDArray[np.int32]]
        List of contours, each as numpy array of shape (N, 1, 2).

    Examples
    --------
    >>> _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    >>> contours = find_contours(binary)
    >>> print(f"Found {len(contours)} contours")
    """
    contours, _ = cv2.findContours(
        binary_image, mode, method
    )  # pylint: disable=no-member
    return [np.asarray(c, dtype=np.int32) for c in contours]


def analyze_contours(
    contours: Sequence[NDArray[np.int32]],
    sort_by: str = "area",
    descending: bool = True,
) -> list[ContourInfo]:
    """Analyze contours and extract geometric information.

    Converts raw contour arrays into ContourInfo objects with
    computed geometric properties.

    Parameters
    ----------
    contours : Sequence[NDArray[np.int32]]
        List of contours to analyze.
    sort_by : str, optional
        Property to sort by: "area", "perimeter", "circularity".
        Default is "area".
    descending : bool, optional
        Sort in descending order (largest first). Default is True.

    Returns
    -------
    list[ContourInfo]
        List of analyzed contours sorted by specified property.

    Examples
    --------
    >>> contours = find_contours(binary)
    >>> analyzed = analyze_contours(contours, sort_by="area")
    >>> largest = analyzed[0]
    """
    contour_info_list = [ContourInfo.from_contour(c) for c in contours]

    # Sort by specified property
    sort_key = {
        "area": lambda c: c.area,
        "perimeter": lambda c: c.perimeter,
        "circularity": lambda c: c.circularity,
    }.get(sort_by, lambda c: c.area)

    contour_info_list.sort(key=sort_key, reverse=descending)
    return contour_info_list


def filter_by_area(
    contours: list[ContourInfo],
    image_area: int,
    min_ratio: float = 0.0002,
    max_ratio: float = 1.0,
) -> list[ContourInfo]:
    """Filter contours by area relative to image size.

    Removes contours that are too small (noise) or too large
    (background) based on image area ratios.

    Parameters
    ----------
    contours : list[ContourInfo]
        List of ContourInfo objects.
    image_area : int
        Total image area in pixels (width * height).
    min_ratio : float, optional
        Minimum contour area as ratio of image area. Default 0.0002.
    max_ratio : float, optional
        Maximum contour area as ratio of image area. Default 1.0.

    Returns
    -------
    list[ContourInfo]
        Filtered list of contours within area bounds.

    Examples
    --------
    >>> h, w = image.shape[:2]
    >>> filtered = filter_by_area(contours, w * h, min_ratio=0.001)
    """
    min_area = image_area * min_ratio
    max_area = image_area * max_ratio
    return [c for c in contours if min_area <= c.area <= max_area]


def filter_by_circularity(
    contours: list[ContourInfo],
    min_circularity: float = 0.0,
    max_circularity: float = 1.0,
) -> list[ContourInfo]:
    """Filter contours by circularity.

    Circularity is defined as 4*pi*area/perimeter^2.
    A perfect circle has circularity of 1.0.

    Parameters
    ----------
    contours : list[ContourInfo]
        List of ContourInfo objects.
    min_circularity : float, optional
        Minimum circularity (0-1). Default 0.0.
    max_circularity : float, optional
        Maximum circularity (0-1). Default 1.0.

    Returns
    -------
    list[ContourInfo]
        Filtered contours within circularity bounds.
    """
    return [
        c
        for c in contours
        if min_circularity <= c.circularity <= max_circularity
    ]


def filter_by_aspect_ratio(
    contours: list[ContourInfo],
    min_ratio: float = 0.0,
    max_ratio: float = float("inf"),
) -> list[ContourInfo]:
    """Filter contours by aspect ratio.

    Parameters
    ----------
    contours : list[ContourInfo]
        List of ContourInfo objects.
    min_ratio : float, optional
        Minimum width/height ratio. Default 0.0.
    max_ratio : float, optional
        Maximum width/height ratio. Default infinity.

    Returns
    -------
    list[ContourInfo]
        Filtered contours within aspect ratio bounds.
    """
    return [c for c in contours if min_ratio <= c.aspect_ratio <= max_ratio]


def get_bounding_boxes(
    contours: list[ContourInfo],
    margin: int = 0,
    image_size: tuple[int, int] | None = None,
) -> list[tuple[int, int, int, int]]:
    """Get bounding boxes for contours with optional margin.

    Parameters
    ----------
    contours : list[ContourInfo]
        List of ContourInfo objects.
    margin : int, optional
        Margin to add around boxes in pixels. Default 0.
    image_size : tuple[int, int] | None, optional
        Image dimensions (height, width) to clamp boxes. If None,
        boxes may extend beyond original bounds.

    Returns
    -------
    list[tuple[int, int, int, int]]
        List of bounding boxes as (x, y, width, height).

    Examples
    --------
    >>> boxes = get_bounding_boxes(contours, margin=10, image_size=image.shape[:2])
    """
    boxes = []
    for c in contours:
        x, y, w, h = c.bounding_rect
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = x + w + margin
        y2 = y + h + margin

        if image_size is not None:
            img_h, img_w = image_size
            x2 = min(img_w, x2)
            y2 = min(img_h, y2)

        boxes.append((x1, y1, x2 - x1, y2 - y1))
    return boxes


def get_rotated_boxes(
    contours: list[ContourInfo],
) -> list[NDArray[np.int32]]:
    """Get minimum area rotated bounding boxes.

    Returns the four corner vertices of the minimum area
    rotated rectangle for each contour.

    Parameters
    ----------
    contours : list[ContourInfo]
        List of ContourInfo objects.

    Returns
    -------
    list[NDArray[np.int32]]
        List of rotated box vertices as numpy arrays of shape (4, 2).

    Examples
    --------
    >>> boxes = get_rotated_boxes(contours)
    >>> for box in boxes:
    ...     cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
    """
    boxes = []
    for c in contours:
        box = cv2.boxPoints(c.min_area_rect)  # pylint: disable=no-member
        boxes.append(np.int32(box))
    return boxes


def find_largest_contour(
    binary_image: NDArray[np.uint8],
    mode: int = cv2.RETR_EXTERNAL,  # pylint: disable=no-member
) -> ContourInfo | None:
    """Find the largest contour by area in a binary image.

    Convenience function for extracting the dominant object.

    Parameters
    ----------
    binary_image : NDArray[np.uint8]
        Binary (thresholded) image.
    mode : int, optional
        Contour retrieval mode. Default cv2.RETR_EXTERNAL.

    Returns
    -------
    ContourInfo | None
        ContourInfo for the largest contour, or None if no contours found.

    Examples
    --------
    >>> largest = find_largest_contour(mask)
    >>> if largest:
    ...     print(f"Largest area: {largest.area}")
    """
    contours = find_contours(binary_image, mode)
    if not contours:
        return None
    analyzed = analyze_contours(contours, sort_by="area", descending=True)
    return analyzed[0] if analyzed else None


def approximate_contour(
    contour: NDArray[np.int32],
    epsilon: float | None = None,
    epsilon_percent: float = 0.02,
    closed: bool = True,
) -> NDArray[np.int32]:
    """Approximate a contour with fewer points.

    Uses Douglas-Peucker algorithm to reduce contour complexity
    while preserving shape.

    Parameters
    ----------
    contour : NDArray[np.int32]
        Input contour.
    epsilon : float | None, optional
        Approximation accuracy (maximum distance from contour).
        If None, calculated from epsilon_percent.
    epsilon_percent : float, optional
        Epsilon as percentage of contour perimeter. Default 0.02 (2%).
    closed : bool, optional
        Whether the contour is closed. Default True.

    Returns
    -------
    NDArray[np.int32]
        Approximated contour with fewer points.

    Examples
    --------
    >>> simplified = approximate_contour(contour, epsilon_percent=0.01)
    >>> print(f"Reduced from {len(contour)} to {len(simplified)} points")
    """
    if epsilon is None:
        perimeter = cv2.arcLength(contour, closed)  # pylint: disable=no-member
        epsilon = epsilon_percent * perimeter
    return np.asarray(
        cv2.approxPolyDP(contour, epsilon, closed), dtype=np.int32
    )  # pylint: disable=no-member


def scale_contour(
    contour: NDArray[np.int32],
    scale: float,
    offset: tuple[int, int] = (0, 0),
) -> NDArray[np.int32]:
    """Scale and offset a contour.

    Useful for transforming contours to different coordinate systems.

    Parameters
    ----------
    contour : NDArray[np.int32]
        Input contour.
    scale : float
        Scale factor (>1 enlarges, <1 shrinks).
    offset : tuple[int, int], optional
        (x, y) offset to subtract after scaling. Default (0, 0).

    Returns
    -------
    NDArray[np.int32]
        Scaled and offset contour.

    Examples
    --------
    >>> # Scale to half size and offset by crop region
    >>> scaled = scale_contour(contour, 0.5, offset=(crop_x, crop_y))
    """
    scaled = contour.astype(np.float64) * scale
    scaled = scaled - np.array([offset[0], offset[1]])
    return scaled.astype(np.int32)


def contours_to_mask(
    contours: list[ContourInfo],
    image_size: tuple[int, int],
    filled: bool = True,
) -> NDArray[np.uint8]:
    """Create a binary mask from contours.

    Parameters
    ----------
    contours : list[ContourInfo]
        List of ContourInfo objects.
    image_size : tuple[int, int]
        Output mask size as (height, width).
    filled : bool, optional
        If True, fill contour interiors. If False, draw outlines only.
        Default True.

    Returns
    -------
    NDArray[np.uint8]
        Binary mask (255 for contour regions, 0 elsewhere).

    Examples
    --------
    >>> mask = contours_to_mask(contours, image.shape[:2])
    >>> result = cv2.bitwise_and(image, image, mask=mask)
    """
    mask = np.zeros(image_size, dtype=np.uint8)
    thickness = -1 if filled else 1
    raw_contours = [c.contour for c in contours]
    cv2.drawContours(
        mask, raw_contours, -1, 255, thickness
    )  # pylint: disable=no-member
    return mask
