"""
Dreamstack Raster - Format Queries
==================================

Functions for querying format support.

"""

from __future__ import annotations

from dreamstack.raster.io.formats.constants import (
    ALPHA_FORMATS,
    HIGH_BIT_DEPTH_FORMATS,
    LAYER_FORMATS,
    READ_FORMATS,
    WRITE_FORMATS,
)
from dreamstack.raster.io.formats.image_format import ImageFormat


def get_supported_formats() -> set[ImageFormat]:
    """Get all supported formats."""
    return READ_FORMATS | WRITE_FORMATS


def get_read_formats() -> set[ImageFormat]:
    """Get formats that can be read."""
    return READ_FORMATS.copy()


def get_write_formats() -> set[ImageFormat]:
    """Get formats that can be written."""
    return WRITE_FORMATS.copy()


def supports_layers(format: ImageFormat) -> bool:
    """Check if format supports layers."""
    return format in LAYER_FORMATS


def supports_alpha(format: ImageFormat) -> bool:
    """Check if format supports alpha channel."""
    return format in ALPHA_FORMATS


def supports_high_bit_depth(format: ImageFormat) -> bool:
    """Check if format supports 16-bit or higher."""
    return format in HIGH_BIT_DEPTH_FORMATS
