# -*- coding: utf-8 -*-

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
    normalize_depth,
    save_depth_image,
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
    "colorize_depth",
    "save_depth_image",
]
