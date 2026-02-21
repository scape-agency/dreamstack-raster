# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Save PSD
============================

Save image as PSD file.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.image import Image


def save_psd(
    image: Image,
    path: str | Path,
    layers: list | None = None,  # pylint: disable=unused-argument  # TODO
    **options,  # pylint: disable=unused-argument
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
