# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Validation Constants
====================

Supported image file extensions.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

# Default supported image extensions
SUPPORTED_EXTENSIONS: tuple[str, ...] = (
    "jpg",
    "jpeg",
    "png",
    "gif",
    "bmp",
    "tiff",
    "tif",
    "webp",
    "ico",
    "ppm",
    "pgm",
    "pbm",
    "hdr",
    "exr",
)

SUPPORTED_RAW_EXTENSIONS: tuple[str, ...] = (
    "raw",
    "cr2",
    "cr3",
    "nef",
    "arw",
    "dng",
    "orf",
    "rw2",
)
