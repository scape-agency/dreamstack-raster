# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Ensure Directory
================

Ensure a directory exists, creating if necessary.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from pathlib import Path


def ensure_directory(path: str | Path) -> Path:
    """Ensure a directory exists, creating if necessary.

    Args:
        path: Directory path.

    Returns:
        Path object (guaranteed to exist).

    Example:
        >>> output_dir = ensure_directory("output/processed")
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
