"""
Selection Operations
====================

Operations for modifying and manipulating selections.

"""

# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from dreamstack.raster.selection.operations.border import border
from dreamstack.raster.selection.operations.contract import contract
from dreamstack.raster.selection.operations.expand import expand
from dreamstack.raster.selection.operations.feather import feather
from dreamstack.raster.selection.operations.grow import grow
from dreamstack.raster.selection.operations.invert import invert
from dreamstack.raster.selection.operations.refine_edge import refine_edge
from dreamstack.raster.selection.operations.similar import similar
from dreamstack.raster.selection.operations.smooth import smooth

__all__: list[str] = [
    "expand",
    "contract",
    "feather",
    "smooth",
    "border",
    "invert",
    "grow",
    "similar",
    "refine_edge",
]
