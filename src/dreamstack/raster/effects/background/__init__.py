# -*- coding: utf-8 -*-

"""
Dreamstack Raster - Background Removal Module
=============================================

Background removal, alpha mask extraction, and compositing utilities.
Uses rembg for AI-based background removal when available.

"""

from __future__ import annotations

from dreamstack.raster.effects.background.removal import (
    composite_on_background,
    create_color_background,
    extract_alpha_mask,
    refine_mask,
    remove_background,
)
from dreamstack.raster.effects.background.replace import (
    replace_background,
    replace_background_with_blur,
    replace_background_with_gradient,
)

__all__: list[str] = [
    # Removal
    "remove_background",
    "extract_alpha_mask",
    "refine_mask",
    # Compositing
    "composite_on_background",
    "create_color_background",
    # Replacement
    "replace_background",
    "replace_background_with_blur",
    "replace_background_with_gradient",
]
