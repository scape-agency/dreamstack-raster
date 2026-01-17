# -*- coding: utf-8 -*-

"""
Dreamstack Raster - Effects Module
==================================

Layer effects: shadows, glows, bevels, and overlays.

"""

from __future__ import annotations

from dreamstack.raster.effects.bevel import BevelStyle, bevel_emboss
from dreamstack.raster.effects.glow import inner_glow, outer_glow
from dreamstack.raster.effects.overlay import (
    color_overlay,
    gradient_overlay,
    pattern_overlay,
    stroke_effect,
)
from dreamstack.raster.effects.shadow import drop_shadow, inner_shadow

__all__: list[str] = [
    # Shadow
    "drop_shadow",
    "inner_shadow",
    # Glow
    "outer_glow",
    "inner_glow",
    # Bevel
    "bevel_emboss",
    "BevelStyle",
    # Overlay
    "color_overlay",
    "gradient_overlay",
    "pattern_overlay",
    "stroke_effect",
]
