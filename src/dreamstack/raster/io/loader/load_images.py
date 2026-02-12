"""
Dreamstack Raster - Load Images
===============================

Load multiple images.

"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dreamstack.raster.core.image import Image


def load_images(paths: list[str | Path], **options) -> list[Image]:
    """
    Load multiple images.

    Args:
        paths: List of image paths
        **options: Loading options

    Returns:
        List of loaded images
    """
    from dreamstack.raster.io.loader.load_image import load_image

    return [load_image(p, **options) for p in paths]
