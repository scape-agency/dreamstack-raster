# -*- coding: utf-8 -*-

"""
Dreamstack Raster - Extraction Module
=====================================

Object extraction from images using contour detection,
preprocessing, and background masking.

This module provides the primary extraction functionality
for isolating objects from scanned images, photographs,
and other visual sources.

"""

from __future__ import annotations

from dreamstack.raster.extraction.extractor import (
    ExtractedObject,
    ExtractionConfig,
    ObjectExtractor,
)
from dreamstack.raster.extraction.operations import (
    apply_background_mask,
    extract_object,
    extract_objects,
    extract_region,
)

__all__: list[str] = [
    # Data Classes
    "ExtractedObject",
    "ExtractionConfig",
    # Extractor Class
    "ObjectExtractor",
    # Operations
    "extract_object",
    "extract_objects",
    "extract_region",
    "apply_background_mask",
]
