"""
Find Images Utility
===================

Find all image files in a directory.
"""

from __future__ import annotations

from pathlib import Path

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def find_images(directory: Path, recursive: bool = True) -> list[Path]:
    """Find all image files in directory.

    Parameters
    ----------
    directory : Path
        Directory to search.
    recursive : bool
        Whether to search recursively. Default True.

    Returns
    -------
    list[Path]
        List of image file paths, sorted.
    """
    pattern = "**/*" if recursive else "*"
    images = []
    for path in directory.glob(pattern):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)
    return sorted(images)
