# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Drawing Module
==================================

Drawing tools: brushes, shapes, text, and vector paths.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from dreamstack.raster.drawing.brush import (
    Brush,
    BrushPreset,
    erase,
    fill_color,
    paint,
    stroke,
)
from dreamstack.raster.drawing.gradient import (
    GradientStop,
    angular_gradient,
    diamond_gradient,
    linear_gradient,
    radial_gradient,
)
from dreamstack.raster.drawing.patterns import (
    PatternConfig,
    checker,
    checker_fast,
    constant,
    gaussian_noise,
    grid,
    noise,
    perlin_noise,
    solid,
    stripes,
    transparency_checker,
)
from dreamstack.raster.drawing.shapes import (
    arrow,
    ellipse,
    line,
    polygon,
    rectangle,
    rounded_rect,
    star,
)
from dreamstack.raster.drawing.text import (
    FontWeight,
    TextStyle,
    draw_text,
    text_bounds,
)

__all__: list[str] = [
    # Brush
    "Brush",
    "BrushPreset",
    "stroke",
    "paint",
    "fill_color",
    "erase",
    # Shapes
    "line",
    "rectangle",
    "ellipse",
    "polygon",
    "rounded_rect",
    "arrow",
    "star",
    # Text
    "draw_text",
    "TextStyle",
    "FontWeight",
    "text_bounds",
    # Gradient
    "linear_gradient",
    "radial_gradient",
    "angular_gradient",
    "diamond_gradient",
    "GradientStop",
    # Patterns
    "constant",
    "solid",
    "checker",
    "checker_fast",
    "transparency_checker",
    "noise",
    "gaussian_noise",
    "perlin_noise",
    "stripes",
    "grid",
    "PatternConfig",
]
