# -*- coding: utf-8 -*-

"""
Dreamstack Raster - Save PSD
============================

Save image as PSD file.

"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from dreamstack.raster.core.image import Image


def save_psd(
    image: Image, path: str | Path, layers: Optional[List] = None, **options
) -> None:
    """
    Save image as PSD file.

    Args:
        image: Image to save (used if no layers specified)
        path: Output path
        layers: Optional list of layers to include
        **options: Additional options
    """
    path = Path(path)

    # Simple PSD save using psd-tools
    pil_image = image.to_pil()

    if pil_image.mode not in ("RGB", "RGBA"):
        pil_image = pil_image.convert("RGBA")

    # psd-tools doesn't have great write support
    # Fall back to PIL which can write basic PSD
    pil_image.save(path, "PSD")
