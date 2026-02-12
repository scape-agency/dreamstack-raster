# -*- coding: utf-8 -*-

"""
Contour Information Data Class
==============================

Data structure for storing contour geometric information.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass
class ContourInfo:
    """Information about a detected contour.

    Holds geometric properties computed from a contour including
    area, bounding rectangles, convexity, and center point.

    Attributes
    ----------
    contour : NDArray[np.int32]
        The contour points as a numpy array of shape (N, 1, 2).
    is_convex : bool
        Whether the contour is convex (no concave regions).
    area : float
        The contour area in pixels (using Green's formula).
    bounding_rect : tuple[int, int, int, int]
        Axis-aligned bounding rectangle as (x, y, width, height).
    min_area_rect : tuple[tuple[float, float], tuple[float, float], float]
        Minimum area rotated rectangle as ((cx, cy), (w, h), angle).
    perimeter : float
        Arc length of the contour.
    circularity : float
        How circular the shape is (1.0 = perfect circle).
    aspect_ratio : float
        Ratio of width to height of bounding rect.

    Examples
    --------
    >>> info = ContourInfo.from_contour(contour)
    >>> print(f"Area: {info.area}, Center: {info.center}")
    """

    contour: NDArray[np.int32]
    is_convex: bool
    area: float
    bounding_rect: Tuple[int, int, int, int]
    min_area_rect: Tuple[Tuple[float, float], Tuple[float, float], float]
    perimeter: float = 0.0
    circularity: float = 0.0
    aspect_ratio: float = 1.0

    @classmethod
    def from_contour(cls, contour: NDArray[np.int32]) -> "ContourInfo":
        """Create ContourInfo from a contour array.

        Parameters
        ----------
        contour : NDArray[np.int32]
            Contour points from cv2.findContours.

        Returns
        -------
        ContourInfo
            Populated contour information object.
        """
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, closed=True)
        bounding_rect = cv2.boundingRect(contour)
        x, y, w, h = bounding_rect

        # Calculate circularity (4 * pi * area / perimeter^2)
        circularity = 0.0
        if perimeter > 0:
            circularity = (4 * np.pi * area) / (perimeter * perimeter)

        # Aspect ratio
        aspect_ratio = float(w) / h if h > 0 else 1.0

        return cls(
            contour=contour,
            is_convex=cv2.isContourConvex(contour),
            area=area,
            bounding_rect=bounding_rect,
            min_area_rect=cv2.minAreaRect(contour),
            perimeter=perimeter,
            circularity=circularity,
            aspect_ratio=aspect_ratio,
        )

    @property
    def center(self) -> Tuple[float, float]:
        """Get the center of the bounding rectangle.

        Returns
        -------
        tuple[float, float]
            Center point (x, y).
        """
        x, y, w, h = self.bounding_rect
        return (x + w / 2, y + h / 2)

    @property
    def centroid(self) -> Tuple[float, float]:
        """Get the centroid using image moments.

        Returns
        -------
        tuple[float, float]
            Centroid point (x, y), or center if moments are zero.
        """
        moments = cv2.moments(self.contour)
        if moments["m00"] != 0:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
            return (cx, cy)
        return self.center

    @property
    def convex_hull(self) -> NDArray[np.int32]:
        """Get the convex hull of the contour.

        Returns
        -------
        NDArray[np.int32]
            Convex hull points.
        """
        return cv2.convexHull(self.contour)

    @property
    def solidity(self) -> float:
        """Calculate solidity (area / convex hull area).

        Returns
        -------
        float
            Solidity ratio (0-1).
        """
        hull = self.convex_hull
        hull_area = cv2.contourArea(hull)
        return self.area / hull_area if hull_area > 0 else 0.0

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ContourInfo(area={self.area:.1f}, "
            f"center={self.center}, "
            f"circularity={self.circularity:.2f})"
        )
