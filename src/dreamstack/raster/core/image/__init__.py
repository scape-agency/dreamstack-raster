# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Image Module
================================

Core Image class and metadata for raster image manipulation.

"""

from dreamstack.raster.core.image.image import Image
from dreamstack.raster.core.image.image_metadata import ImageMetadata

__all__: list[str] = [
    "ImageMetadata",
    "Image",
]
