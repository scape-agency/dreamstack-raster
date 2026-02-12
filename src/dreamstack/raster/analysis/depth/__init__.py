"""
Dreamstack Raster - Depth Estimation Module
============================================

AI-based monocular depth estimation using transformer models.
Supports Depth Anything and other depth estimation models.

"""

from __future__ import annotations

from dreamstack.raster.analysis.depth.estimator import (
    DepthConfig,
    DepthEstimator,
    DepthResult,
)
from dreamstack.raster.analysis.depth.operations import (
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

__all__: list[str] = [
    # Classes
    "DepthEstimator",
    "DepthConfig",
    "DepthResult",
    # Operations
    "estimate_depth",
    "estimate_depth_batch",
    "normalize_depth",
    "normalize_depth_advanced",
    "colorize_depth",
    "save_depth_image",
    "save_depth_ply",
    "save_depth_npy",
    "load_depth_npy",
]
