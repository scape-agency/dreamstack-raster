# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Smart Image Compression
=======================

Utilities for intelligent image compression with target file size.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from .compress_image import compress_image
from .compress_to_size import compress_to_size
from .compression_config import CompressionConfig, CompressionFormat
from .compression_result import CompressionResult
from .estimate_file_size import estimate_file_size
from .optimize_for_web import optimize_for_web

__all__ = [
    "CompressionConfig",
    "CompressionFormat",
    "CompressionResult",
    "compress_to_size",
    "compress_image",
    "estimate_file_size",
    "optimize_for_web",
]
