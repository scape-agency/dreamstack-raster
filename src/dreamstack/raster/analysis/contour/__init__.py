# -*- coding: utf-8 -*-

"""
Dreamstack Raster - Contour Analysis Module
==========================================

Contour detection and geometric analysis for shape identification
and object extraction from images.

"""

from __future__ import annotations

from dreamstack.raster.analysis.contour.active_contour import (
    ActiveContourConfig,
    ActiveContourResult,
    active_contour,
    contour_area,
    contour_centroid,
    contour_perimeter,
    contour_to_mask,
    create_circular_contour,
    create_elliptical_contour,
    create_rectangular_contour,
    draw_contour,
    extract_contour_region,
)
from dreamstack.raster.analysis.contour.detector import (
    ContourDetector,
    DetectionConfig,
)
from dreamstack.raster.analysis.contour.info import ContourInfo
from dreamstack.raster.analysis.contour.operations import (
    analyze_contours,
    approximate_contour,
    filter_by_area,
    find_contours,
    find_largest_contour,
    get_bounding_boxes,
    get_rotated_boxes,
    scale_contour,
)

__all__: list[str] = [
    # Data Classes
    "ContourInfo",
    "DetectionConfig",
    "ActiveContourConfig",
    "ActiveContourResult",
    # Detector Class
    "ContourDetector",
    # Operations
    "find_contours",
    "analyze_contours",
    "filter_by_area",
    "get_bounding_boxes",
    "get_rotated_boxes",
    "find_largest_contour",
    "approximate_contour",
    "scale_contour",
    # Active Contour (Snakes)
    "active_contour",
    "create_circular_contour",
    "create_elliptical_contour",
    "create_rectangular_contour",
    "draw_contour",
    "contour_to_mask",
    "extract_contour_region",
    "contour_area",
    "contour_perimeter",
    "contour_centroid",
]
