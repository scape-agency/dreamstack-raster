"""
Brush Tools
===========

Brush-based drawing operations.

"""

from __future__ import annotations

from dreamstack.raster.drawing.brush.brush import Brush
from dreamstack.raster.drawing.brush.brush_preset import BrushPreset
from dreamstack.raster.drawing.brush.erase import erase
from dreamstack.raster.drawing.brush.fill_color import fill_color
from dreamstack.raster.drawing.brush.paint import paint
from dreamstack.raster.drawing.brush.stroke import stroke

__all__: list[str] = [
    "Brush",
    "BrushPreset",
    "stroke",
    "paint",
    "fill_color",
    "erase",
]
