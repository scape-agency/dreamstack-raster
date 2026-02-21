"""
Color-Based Selections
======================

Selection tools based on color and image analysis.

"""

# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from dreamstack.raster.selection.color.color_range import color_range
from dreamstack.raster.selection.color.magic_wand import magic_wand
from dreamstack.raster.selection.color.select_focus import select_focus
from dreamstack.raster.selection.color.select_subject import select_subject

__all__: list[str] = [
    "magic_wand",
    "color_range",
    "select_subject",
    "select_focus",
]
