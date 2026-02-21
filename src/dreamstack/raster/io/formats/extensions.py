# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Format Extensions
=====================================

Functions for getting file extensions for formats.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from dreamstack.raster.io.formats.constants import FORMAT_EXTENSIONS
from dreamstack.raster.io.formats.image_format import ImageFormat


def get_extensions(image_format: ImageFormat) -> list[str]:
    """
    Get file extensions for a format.

    Args:
        image_format: Image format

    Returns:
        List of extensions (with dots)
    """
    return FORMAT_EXTENSIONS.get(image_format, [])


def get_primary_extension(image_format: ImageFormat) -> str:
    """
    Get primary extension for a format.

    Args:
        image_format: Image format

    Returns:
        Primary extension (with dot)
    """
    extensions = FORMAT_EXTENSIONS.get(image_format, [])
    return extensions[0] if extensions else ""
