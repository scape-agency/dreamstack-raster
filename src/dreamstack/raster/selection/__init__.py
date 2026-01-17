# -*- coding: utf-8 -*-

"""
Dreamstack Raster - Selection Module
====================================

Selection tools: rectangular, elliptical, lasso, polygon,
magic wand, color range, and selection operations.

"""

from __future__ import annotations

from dreamstack.raster.selection.color import (
    color_range,
    magic_wand,
    select_focus,
    select_subject,
)
from dreamstack.raster.selection.freeform import (
    lasso,
    magnetic_lasso,
    polygon,
    quick_selection,
)
from dreamstack.raster.selection.operations import (
    border,
    contract,
    expand,
    feather,
    grow,
    invert,
    refine_edge,
    similar,
    smooth,
)
from dreamstack.raster.selection.shapes import (
    Selection,
    SelectionMode,
    elliptical,
    rectangular,
    rounded_rectangle,
    single_column,
    single_row,
)

__all__: list[str] = [
    # Core
    "Selection",
    "SelectionMode",
    # Shapes
    "rectangular",
    "elliptical",
    "rounded_rectangle",
    "single_row",
    "single_column",
    # Freeform
    "lasso",
    "polygon",
    "magnetic_lasso",
    "quick_selection",
    # Color-based
    "magic_wand",
    "color_range",
    "select_subject",
    "select_focus",
    # Operations
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
