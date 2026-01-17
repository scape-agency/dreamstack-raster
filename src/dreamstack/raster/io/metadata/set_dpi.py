# -*- coding: utf-8 -*-

"""
Dreamstack Raster - Set DPI
===========================

Set DPI/resolution metadata for image files.

"""

from __future__ import annotations

from pathlib import Path


def set_dpi(path: str | Path, dpi: tuple[float, float]) -> None:
    """
    Set DPI/resolution metadata.

    Args:
        path: Path to image file
        dpi: DPI as (x, y) tuple
    """
    from PIL import Image as PILImage

    path = Path(path)

    with PILImage.open(path) as img:
        img.save(path, dpi=dpi)
