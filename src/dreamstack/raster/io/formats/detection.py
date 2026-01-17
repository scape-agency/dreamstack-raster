# -*- coding: utf-8 -*-

"""
Dreamstack Raster - Format Detection
====================================

Functions for detecting image format from path or MIME type.

"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from dreamstack.raster.io.formats.constants import (
    FORMAT_EXTENSIONS,
    FORMAT_MIME_TYPES,
)
from dreamstack.raster.io.formats.image_format import ImageFormat


def get_format_for_path(path: str | Path) -> Optional[ImageFormat]:
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


def get_format_for_mime(mime_type: str) -> Optional[ImageFormat]:
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
