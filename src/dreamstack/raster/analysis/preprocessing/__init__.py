# -*- coding: utf-8 -*-

"""
Dreamstack Raster - Preprocessing Module
========================================

Image preprocessing pipelines for object detection,
segmentation, and extraction workflows.

"""

from __future__ import annotations

from dreamstack.raster.analysis.preprocessing.processor import (
    ImagePreprocessor,
    PreprocessingConfig,
)
from dreamstack.raster.analysis.preprocessing.operations import (
    apply_clahe,
    binarize,
    detect_edges,
    morphological_close,
    morphological_open,
    preprocess_for_contours,
    to_grayscale,
)

__all__: list[str] = [
    # Configuration and Processor
    "PreprocessingConfig",
    "ImagePreprocessor",
    # Operations
    "to_grayscale",
    "apply_clahe",
    "binarize",
    "detect_edges",
    "morphological_open",
    "morphological_close",
    "preprocess_for_contours",
]
