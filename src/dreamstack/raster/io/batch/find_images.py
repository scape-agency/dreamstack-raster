# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - find_images
===========

Find all image files in a directory.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from pathlib import Path


def find_images(
    source: str | Path,
    *,
    extensions: tuple[str, ...] | None = None,
    recursive: bool = False,
) -> list[Path]:
    """Find all image files in a directory.

    Args:
        source: Directory path or single file.
        extensions: Allowed extensions (default: common formats).
        recursive: Search subdirectories.

    Returns:
        List of image file paths.

    Example:
        >>> images = find_images("/photos", recursive=True)
        >>> print(f"Found {len(images)} images")
    """
    from dreamstack.raster.io.validation import (
        SUPPORTED_EXTENSIONS,
        get_image_files,
    )

    source = Path(source)

    if source.is_file():
        return [source]

    exts = extensions if extensions else SUPPORTED_EXTENSIONS

    return get_image_files(source, extensions=exts, recursive=recursive)
