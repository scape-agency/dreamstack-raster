# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Edge Detection Filters
==========================================

Edge detection and enhancement filters.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from dreamstack.raster.filters.edge.canny import canny, canny_auto
from dreamstack.raster.filters.edge.edge_detect import edge_detect
from dreamstack.raster.filters.edge.emboss import emboss
from dreamstack.raster.filters.edge.find_edges import find_edges
from dreamstack.raster.filters.edge.gradient_direction import (
    gradient_direction,
)
from dreamstack.raster.filters.edge.laplacian import laplacian
from dreamstack.raster.filters.edge.prewitt import prewitt
from dreamstack.raster.filters.edge.scharr import scharr
from dreamstack.raster.filters.edge.sobel import sobel

__all__: list[str] = [
    "edge_detect",
    "sobel",
    "canny",
    "canny_auto",
    "laplacian",
    "prewitt",
    "scharr",
    "find_edges",
    "emboss",
    "gradient_direction",
]
