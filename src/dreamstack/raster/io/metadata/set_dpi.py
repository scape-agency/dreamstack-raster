# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Set DPI
===========================

Set DPI/resolution metadata for image files.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from pathlib import Path


def set_dpi(path: str | Path, dpi: tuple[float, float]) -> None:
    """
    Set DPI/resolution metadata.

    Args:
        path: Path to image file
        dpi: DPI as (x, y) tuple
    """
    # pylint: disable=import-outside-toplevel
    from PIL import Image as PILImage

    path = Path(path)

    with PILImage.open(path) as img:
        img.save(path, dpi=dpi)
