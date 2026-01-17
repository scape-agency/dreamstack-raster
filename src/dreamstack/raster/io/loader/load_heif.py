# -*- coding: utf-8 -*-

"""
Dreamstack Raster - Load HEIF
=============================

Load HEIF/HEIC image.

"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dreamstack.raster.core.image import Image


def load_heif(path: Path, **options) -> Image:
    """Load HEIF/HEIC image."""
    from pillow_heif import register_heif_opener

    from dreamstack.raster.io.loader.load_with_pil import load_with_pil

    register_heif_opener()

    return load_with_pil(path, **options)
