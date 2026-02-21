# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Load HEIF
=============================

Load HEIF/HEIC image.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

# =============================================================================
# Type Checking Imports
# =============================================================================

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.image import Image


def load_heif(path: Path, **options) -> Image:
    """Load HEIF/HEIC image."""
    # pylint: disable=import-outside-toplevel
    from pillow_heif import register_heif_opener

    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.io.loader.load_with_pil import load_with_pil

    register_heif_opener()

    return load_with_pil(path, **options)
