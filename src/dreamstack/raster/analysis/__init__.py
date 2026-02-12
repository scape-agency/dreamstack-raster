# -*- coding: utf-8 -*-

"""
Dreamstack Raster - Analysis Module
===================================

Image analysis: histogram, statistics, color picker, contour detection,
preprocessing, and color analysis.

"""

from __future__ import annotations

# Color analysis
from dreamstack.raster.analysis.analysis import (
    ColorAnalyzer,
    adjust_background_color,
    create_gradient_background,
    find_background_color,
    get_dominant_color,
    get_dominant_colors,
    get_most_common_color,
)

# Contour detection
from dreamstack.raster.analysis.contour import (
    ActiveContourConfig,
    ActiveContourResult,
    ContourDetector,
    ContourInfo,
    DetectionConfig,
    active_contour,
    analyze_contours,
    approximate_contour,
    contour_area,
    contour_centroid,
    contour_perimeter,
    contour_to_mask,
    create_circular_contour,
    create_elliptical_contour,
    create_rectangular_contour,
    draw_contour,
    extract_contour_region,
    filter_by_area,
    find_contours,
    find_largest_contour,
    get_bounding_boxes,
    get_rotated_boxes,
    scale_contour,
)

# Depth estimation
from dreamstack.raster.analysis.depth import (
    DepthConfig,
    DepthEstimator,
    DepthResult,
    colorize_depth,
    estimate_depth,
    estimate_depth_batch,
    load_depth_npy,
    normalize_depth,
    normalize_depth_advanced,
    save_depth_image,
    save_depth_npy,
    save_depth_ply,
)

# Face detection and alignment
from dreamstack.raster.analysis.face import (
    AlignmentResult,
    FaceBbox,
    align_eyes,
    detect_face,
    detect_faces,
    detect_landmarks,
    normalize_face_scale,
)
from dreamstack.raster.analysis.histogram import (
    cumulative_histogram,
    histogram,
    histogram_luminosity,
    histogram_rgb,
    histogram_stats,
)

# Template matching
from dreamstack.raster.analysis.matching import (
    MatchMethod,
    MatchResult,
    MultiMatchResult,
    create_template_mask,
    draw_matches,
    find_pattern,
    highlight_match,
    match_template,
    match_template_multi,
)
from dreamstack.raster.analysis.measure import (
    color_sampler,
    measure_selection,
    pixel_info,
    sample_color,
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
from dreamstack.raster.analysis.statistics import (
    channel_statistics,
    color_count,
    image_statistics,
    unique_colors,
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
    # Active Contour (Snakes)
    "ActiveContourConfig",
    "ActiveContourResult",
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
    # Template Matching
    "MatchMethod",
    "MatchResult",
    "MultiMatchResult",
    "match_template",
    "match_template_multi",
    "draw_matches",
    "highlight_match",
    "create_template_mask",
    "find_pattern",
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
    "normalize_depth_advanced",
    "colorize_depth",
    "save_depth_image",
    "save_depth_ply",
    "save_depth_npy",
    "load_depth_npy",
    # Face Detection & Alignment
    "FaceBbox",
    "FaceLandmarks",
    "AlignmentResult",
    "detect_face",
    "detect_faces",
    "detect_landmarks",
    "crop_face",
    "align_eyes",
    "normalize_face_scale",
    "compute_inverse_transform",
    "apply_transform",
]
