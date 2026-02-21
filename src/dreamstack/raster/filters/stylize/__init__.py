# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Stylize Filters
===================================

Stylization and special effect filters.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from dreamstack.raster.filters.stylize.contour import contour
from dreamstack.raster.filters.stylize.diffuse import diffuse
from dreamstack.raster.filters.stylize.extrude import extrude
from dreamstack.raster.filters.stylize.find_edges import find_edges
from dreamstack.raster.filters.stylize.pixelate import (
    PixelateConfig,
    match_to_palette,
    mosaic,
    pixelate,
    pixelate_and_quantize,
    posterize,
    quantize_colors,
)
from dreamstack.raster.filters.stylize.solarize import solarize
from dreamstack.raster.filters.stylize.tiles import tiles
from dreamstack.raster.filters.stylize.trace_contour import trace_contour
from dreamstack.raster.filters.stylize.wind import wind

__all__: list[str] = [
    "diffuse",
    "find_edges",
    "solarize",
    "tiles",
    "extrude",
    "wind",
    "contour",
    "trace_contour",
    # Pixelation
    "pixelate",
    "mosaic",
    "quantize_colors",
    "match_to_palette",
    "posterize",
    "pixelate_and_quantize",
    "PixelateConfig",
]
