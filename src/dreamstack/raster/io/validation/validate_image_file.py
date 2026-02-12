"""
Image File Validation
=====================

Validate that a path is an existing image file.

"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from dreamstack.raster.io.validation.validate_file import validate_file
from dreamstack.raster.io.validation.validate_image_extension import (
    validate_image_extension,
)


def validate_image_file(
    path: str | Path,
    *,
    extensions: Sequence[str] | None = None,
    include_raw: bool = False,
) -> Path:
    """Validate that a path is an existing image file.

    Combines file existence and extension validation.

    Args:
        path: Path to the image file.
        extensions: Custom allowed extensions.
        include_raw: Include RAW formats.

    Returns:
        Validated Path object.

    Example:
        >>> img = validate_image_file("photo.jpg")
    """
    path = validate_file(path)
    path = validate_image_extension(
        path, extensions=extensions, include_raw=include_raw
    )
    return path
