# -*- coding: utf-8 -*-

"""
Dreamstack Raster - Strip Metadata
==================================

Remove all metadata from image files.

"""

from __future__ import annotations

from pathlib import Path


def strip_metadata(path: str | Path) -> None:
    """
    Remove all metadata from an image.

    Args:
        path: Path to image file
    """
    from PIL import Image as PILImage

    path = Path(path)

    with PILImage.open(path) as img:
        # Create clean image without metadata
        data = list(img.getdata())
        clean_img = PILImage.new(img.mode, img.size)
        clean_img.putdata(data)
        clean_img.save(path)
