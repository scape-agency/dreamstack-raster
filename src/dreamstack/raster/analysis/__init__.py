# -*- coding: utf-8 -*-

"""
Dreamstack Raster - Analysis Module
===================================

Image analysis: histogram, statistics, color picker, contour detection,
preprocessing, and color analysis.

"""

from __future__ import annotations

from dreamstack.raster.analysis.histogram import (
    cumulative_histogram,
    histogram,
    histogram_luminosity,
    histogram_rgb,
    histogram_stats,
)
from dreamstack.raster.analysis.measure import (
    color_sampler,
    measure_selection,
    pixel_info,
    sample_color,
)
from dreamstack.raster.analysis.statistics import (
    channel_statistics,
    color_count,
    image_statistics,
    unique_colors,
)

# Contour detection
from dreamstack.raster.analysis.contour import (
    ContourDetector,
    ContourInfo,
    DetectionConfig,
    analyze_contours,
    approximate_contour,
    filter_by_area,
    find_contours,
    find_largest_contour,
    get_bounding_boxes,
    get_rotated_boxes,
    scale_contour,
)

# Image preprocessing
from dreamstack.raster.analysis.preprocessing import (
    ImagePreprocessor,
    PreprocessingConfig,
    apply_clahe,
    binarize,
    detect_edges,
    morphological_close,
    morphological_open,
    preprocess_for_contours,
    to_grayscale,
)

# Color analysis
from dreamstack.raster.analysis.coloranalysis import (
    ColorAnalyzer,
    adjust_background_color,
    create_gradient_background,
    find_background_color,
    get_dominant_color,
    get_dominant_colors,
    get_most_common_color,
)

# Depth estimation
from dreamstack.raster.analysis.depth import (
    DepthConfig,
    DepthEstimator,
    DepthResult,
    colorize_depth,
    estimate_depth,
    estimate_depth_batch,
    normalize_depth,
    save_depth_image,
)

__all__: list[str] = [
    # Histogram
    "histogram",
    "histogram_rgb",
    "histogram_luminosity",
    "cumulative_histogram",
    "histogram_stats",
    # Statistics
    "image_statistics",
    "channel_statistics",
    "color_count",
    "unique_colors",
    # Measure
    "measure_selection",
    "pixel_info",
    "sample_color",
    "color_sampler",
    # Contour Detection
    "ContourInfo",
    "DetectionConfig",
    "ContourDetector",
    "find_contours",
    "analyze_contours",
    "filter_by_area",
    "get_bounding_boxes",
    "get_rotated_boxes",
    "find_largest_contour",
    "approximate_contour",
    "scale_contour",
    # Preprocessing
    "PreprocessingConfig",
    "ImagePreprocessor",
    "to_grayscale",
    "apply_clahe",
    "binarize",
    "detect_edges",
    "morphological_open",
    "morphological_close",
    "preprocess_for_contours",
    # Color Analysis
    "ColorAnalyzer",
    "get_dominant_color",
    "get_dominant_colors",
    "get_most_common_color",
    "find_background_color",
    "adjust_background_color",
    "create_gradient_background",
    # Depth Estimation
    "DepthConfig",
    "DepthEstimator",
    "DepthResult",
    "estimate_depth",
    "estimate_depth_batch",
    "normalize_depth",
    "colorize_depth",
    "save_depth_image",
]
