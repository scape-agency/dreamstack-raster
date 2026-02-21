# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Text Drawing
============

Text rendering operations.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from dreamstack.raster.drawing.text.draw_text import (
    draw_text,
    render_text,
    text_bounds,
)
from dreamstack.raster.drawing.text.text_style import FontWeight, TextStyle

__all__: list[str] = [
    "draw_text",
    "render_text",
    "TextStyle",
    "FontWeight",
    "text_bounds",
]
