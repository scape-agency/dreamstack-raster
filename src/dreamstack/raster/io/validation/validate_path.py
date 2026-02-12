"""
Path Validation
===============

Validate that a path exists.

"""

from __future__ import annotations

from pathlib import Path


def validate_path(path: str | Path) -> Path:
    """Validate that a path exists.

    Args:
        path: Path string or Path object.

    Returns:
        Validated Path object.

    Raises:
        FileNotFoundError: If path does not exist.

    Example:
        >>> path = validate_path("/path/to/file.png")
    """
    from dreamstack.raster.core.exceptions import InvalidPathError

    path = Path(path)

    if not path.exists():
        raise InvalidPathError(f"Path does not exist: {path}")

    return path
