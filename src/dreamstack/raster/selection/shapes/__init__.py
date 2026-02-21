"""
Shape-Based Selections
======================

Geometric shape selection tools.

"""

# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from dreamstack.raster.selection.shapes.elliptical import elliptical
from dreamstack.raster.selection.shapes.rectangular import rectangular
from dreamstack.raster.selection.shapes.rounded_rectangle import (
    rounded_rectangle,
)
from dreamstack.raster.selection.shapes.selection import Selection
from dreamstack.raster.selection.shapes.selection_mode import SelectionMode
from dreamstack.raster.selection.shapes.single_column import single_column
from dreamstack.raster.selection.shapes.single_row import single_row

__all__: list[str] = [
    "Selection",
    "SelectionMode",
    "rectangular",
    "elliptical",
    "rounded_rectangle",
    "single_row",
    "single_column",
]
