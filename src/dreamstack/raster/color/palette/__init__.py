"""
Dreamstack Raster - Color Palette
=================================

Color palette extraction and management.

"""

from __future__ import annotations

from dreamstack.raster.color.palette.color import Color
from dreamstack.raster.color.palette.color_presets import (
    BLACK,
    BLUE,
    CYAN,
    GREEN,
    MAGENTA,
    RED,
    TRANSPARENT,
    WHITE,
    YELLOW,
)
from dreamstack.raster.color.palette.create_gradient import create_gradient
from dreamstack.raster.color.palette.extract_palette import (
    _median_cut,
    _octree_quantize,
    extract_palette,
)
from dreamstack.raster.color.palette.palette import Palette

__all__: list[str] = [
    # Classes
    "Color",
    "Palette",
    # Color presets
    "BLACK",
    "WHITE",
    "RED",
    "GREEN",
    "BLUE",
    "YELLOW",
    "CYAN",
    "MAGENTA",
    "TRANSPARENT",
    # Functions
    "extract_palette",
    "create_gradient",
    # Internal helpers (exported for compatibility)
    "_median_cut",
    "_octree_quantize",
]
