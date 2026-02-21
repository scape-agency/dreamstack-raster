"""
Dreamstack Raster - Effects Module
==================================

Layer effects: shadows, glows, bevels, overlays, and background operations.

"""

# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from dreamstack.raster.effects.background import (
    composite_on_background,
    create_color_background,
    extract_alpha_mask,
    refine_mask,
    remove_background,
    replace_background,
    replace_background_with_blur,
    replace_background_with_gradient,
)
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
    # Background
    "remove_background",
    "extract_alpha_mask",
    "refine_mask",
    "composite_on_background",
    "create_color_background",
    "replace_background",
    "replace_background_with_blur",
    "replace_background_with_gradient",
]
