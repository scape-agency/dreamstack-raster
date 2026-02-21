# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - BatchConfig
===========

Configuration dataclass for batch processing operations.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BatchConfig:
    """Configuration for batch processing.

    Attributes:
        max_workers: Maximum parallel workers (None = CPU count).
        recursive: Search directories recursively.
        extensions: Allowed file extensions.
        skip_errors: Continue on errors instead of stopping.
        verbose: Print progress messages.
        overwrite: Overwrite existing output files.
    """

    max_workers: int | None = None
    recursive: bool = False
    extensions: tuple[str, ...] | None = None
    skip_errors: bool = True
    verbose: bool = True
    overwrite: bool = False
