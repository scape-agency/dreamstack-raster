# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Shape Drawing
=============

Geometric shape drawing operations.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from dreamstack.raster.drawing.shapes.arrow import arrow
from dreamstack.raster.drawing.shapes.ellipse import ellipse
from dreamstack.raster.drawing.shapes.line import line
from dreamstack.raster.drawing.shapes.polygon import polygon
from dreamstack.raster.drawing.shapes.rectangle import rectangle
from dreamstack.raster.drawing.shapes.rounded_rect import rounded_rect
from dreamstack.raster.drawing.shapes.star import star

__all__: list[str] = [
    "line",
    "rectangle",
    "ellipse",
    "polygon",
    "rounded_rect",
    "arrow",
    "star",
]
