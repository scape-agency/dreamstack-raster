# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Format Detection
====================================

Functions for detecting image format from path or MIME type.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from pathlib import Path

from dreamstack.raster.io.formats.constants import (
    FORMAT_EXTENSIONS,
    FORMAT_MIME_TYPES,
)
from dreamstack.raster.io.formats.image_format import ImageFormat


def get_format_for_path(path: str | Path) -> ImageFormat | None:
    """
    Detect format from file path extension.

    Args:
        path: File path

    Returns:
        Detected format or None
    """
    path = Path(path)
    ext = path.suffix.lower()

    for fmt, extensions in FORMAT_EXTENSIONS.items():
        if ext in extensions:
            return fmt

    return None


def get_format_for_mime(mime_type: str) -> ImageFormat | None:
    """
    Get format from MIME type.

    Args:
        mime_type: MIME type string

    Returns:
        Corresponding format or None
    """
    for fmt, mime in FORMAT_MIME_TYPES.items():
        if mime == mime_type:
            return fmt
    return None
