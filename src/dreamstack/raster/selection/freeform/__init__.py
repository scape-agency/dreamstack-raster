"""
Freeform Selections
===================

Freehand and polygon-based selection tools.

"""

# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from dreamstack.raster.selection.freeform.lasso import lasso
from dreamstack.raster.selection.freeform.magnetic_lasso import magnetic_lasso
from dreamstack.raster.selection.freeform.polygon import polygon
from dreamstack.raster.selection.freeform.quick_selection import (
    quick_selection,
)

__all__: list[str] = [
    "lasso",
    "polygon",
    "magnetic_lasso",
    "quick_selection",
]
