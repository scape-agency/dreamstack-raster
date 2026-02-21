# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Directory Validation
====================

Validate that a path is an existing directory.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from pathlib import Path

from dreamstack.raster.io.validation.validate_path import validate_path


def validate_directory(path: str | Path) -> Path:
    """Validate that a path is an existing directory.

    Args:
        path: Path to validate.

    Returns:
        Validated Path object.

    Raises:
        FileNotFoundError: If path does not exist.
        DirectoryNotFoundError: If path exists but is not a directory.

    Example:
        >>> dir_path = validate_directory("/path/to/images")
    """
    from dreamstack.raster.core.exceptions import DirectoryNotFoundError

    path = validate_path(path)

    if not path.is_dir():
        raise DirectoryNotFoundError(
            f"Path exists but is not a directory: {path}"
        )

    return path
