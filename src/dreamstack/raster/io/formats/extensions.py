"""
Dreamstack Raster - Format Extensions
=====================================

Functions for getting file extensions for formats.

"""

from __future__ import annotations

from dreamstack.raster.io.formats.constants import FORMAT_EXTENSIONS
from dreamstack.raster.io.formats.image_format import ImageFormat


def get_extensions(format: ImageFormat) -> list[str]:
    """
    Get file extensions for a format.

    Args:
        format: Image format

    Returns:
        List of extensions (with dots)
    """
    return FORMAT_EXTENSIONS.get(format, [])


def get_primary_extension(format: ImageFormat) -> str:
    """
    Get primary extension for a format.

    Args:
        format: Image format

    Returns:
        Primary extension (with dot)
    """
    extensions = FORMAT_EXTENSIONS.get(format, [])
    return extensions[0] if extensions else ""
