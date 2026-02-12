"""
File Validation
===============

Validate that a path is an existing file.

"""

from __future__ import annotations

from pathlib import Path

from dreamstack.raster.io.validation.validate_path import validate_path


def validate_file(path: str | Path) -> Path:
    """Validate that a path is an existing file.

    Args:
        path: Path to validate.

    Returns:
        Validated Path object.

    Raises:
        FileNotFoundError: If path does not exist.
        NotAFileError: If path exists but is not a file.

    Example:
        >>> file_path = validate_file("image.png")
    """
    from dreamstack.raster.core.exceptions import NotAFileError

    path = validate_path(path)

    if not path.is_file():
        raise NotAFileError(f"Path exists but is not a file: {path}")

    return path
