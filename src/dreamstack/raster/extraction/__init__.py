# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Extraction Module
=====================================

Object extraction from images using contour detection,
preprocessing, and background masking.

This module provides the primary extraction functionality
for isolating objects from scanned images, photographs,
and other visual sources.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from dreamstack.raster.extraction.apply_background_mask import (
    apply_background_mask,
)
from dreamstack.raster.extraction.extract_object import extract_object
from dreamstack.raster.extraction.extract_objects import extract_objects
from dreamstack.raster.extraction.extract_region import extract_region
from dreamstack.raster.extraction.extract_with_alpha import extract_with_alpha
from dreamstack.raster.extraction.extracted_object import ExtractedObject
from dreamstack.raster.extraction.extraction_config import ExtractionConfig
from dreamstack.raster.extraction.object_extractor import ObjectExtractor

__all__: list[str] = [
    # Data Classes
    "ExtractedObject",
    "ExtractionConfig",
    # Extractor Class
    "ObjectExtractor",
    # Operations
    "apply_background_mask",
    "extract_object",
    "extract_objects",
    "extract_region",
    "extract_with_alpha",
]
