# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Get Image Files
===============

Get all image files in a directory.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from dreamstack.raster.io.validation.constants import (
    SUPPORTED_EXTENSIONS,
    SUPPORTED_RAW_EXTENSIONS,
)
from dreamstack.raster.io.validation.validate_directory import (
    validate_directory,
)


def get_image_files(
    directory: str | Path,
    *,
    extensions: Sequence[str] | None = None,
    include_raw: bool = False,
    recursive: bool = False,
) -> list[Path]:
    """Get all image files in a directory.

    Args:
        directory: Directory to search.
        extensions: Specific extensions to match.
        include_raw: Include RAW camera formats.
        recursive: Search subdirectories.

    Returns:
        List of image file paths.

    Example:
        >>> images = get_image_files("/photos", recursive=True)
    """
    directory = validate_directory(directory)

    if extensions is None:
        exts = list(SUPPORTED_EXTENSIONS)
        if include_raw:
            exts.extend(SUPPORTED_RAW_EXTENSIONS)
    else:
        exts = [e.lower().lstrip(".") for e in extensions]

    files = []
    search_fn = directory.rglob if recursive else directory.glob

    for ext in exts:
        # Case-insensitive matching
        pattern = f"*.{ext}"
        files.extend(search_fn(pattern))
        files.extend(search_fn(pattern.upper()))

    # Remove duplicates and sort
    return sorted(set(files))
