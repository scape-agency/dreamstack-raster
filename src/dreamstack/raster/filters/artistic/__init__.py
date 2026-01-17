# -*- coding: utf-8 -*-

"""
Dreamstack Raster - Artistic Filters
====================================

Artistic and stylization filters.

"""

from __future__ import annotations

from dreamstack.raster.filters.artistic.cartoon import cartoon
from dreamstack.raster.filters.artistic.glitch import glitch
from dreamstack.raster.filters.artistic.halftone import halftone
from dreamstack.raster.filters.artistic.oil_paint import oil_paint
from dreamstack.raster.filters.artistic.pixelate import pixelate
from dreamstack.raster.filters.artistic.posterize import posterize
from dreamstack.raster.filters.artistic.sketch import sketch
from dreamstack.raster.filters.artistic.stipple import stipple
from dreamstack.raster.filters.artistic.vignette import vignette
from dreamstack.raster.filters.artistic.watercolor import watercolor

__all__: list[str] = [
    "oil_paint",
    "watercolor",
    "posterize",
    "pixelate",
    "halftone",
    "stipple",
    "sketch",
    "cartoon",
    "glitch",
    "vignette",
]
