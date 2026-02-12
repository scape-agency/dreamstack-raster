"""
Image File Check
================

Check if a path is a valid image file (non-raising).

"""

from __future__ import annotations

from pathlib import Path

from dreamstack.raster.io.validation.validate_image_file import (
    validate_image_file,
)


def is_valid_image_file(path: str | Path) -> bool:
    """Check if a path is a valid image file.

    Non-raising version of validate_image_file.

    Args:
        path: Path to check.

    Returns:
        True if valid image file, False otherwise.

    Example:
        >>> if is_valid_image_file("photo.jpg"):
        ...     process_image("photo.jpg")
    """
    try:
        validate_image_file(path)
        return True
    except Exception:
        return False
