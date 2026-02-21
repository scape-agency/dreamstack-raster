# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Background Removal Module
=============================================

Background removal, alpha mask extraction, and compositing utilities.
Uses rembg for AI-based background removal when available.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

# Compositing functions
from dreamstack.raster.effects.background.composite_on_background import (
    composite_on_background,
)
from dreamstack.raster.effects.background.create_color_background import (
    create_color_background,
)
from dreamstack.raster.effects.background.extract_alpha_mask import (
    extract_alpha_mask,
)
from dreamstack.raster.effects.background.gradient_config import (
    GradientConfig,
    GradientDirection,
)
from dreamstack.raster.effects.background.mask_refinement_config import (
    MaskRefinementConfig,
)
from dreamstack.raster.effects.background.refine_mask import refine_mask

# Config classes
from dreamstack.raster.effects.background.removal_config import (
    ModelName,
    RemovalConfig,
)

# Removal functions
from dreamstack.raster.effects.background.remove_background import (
    remove_background,
)

# Replacement functions
from dreamstack.raster.effects.background.replace_background import (
    replace_background,
)
from dreamstack.raster.effects.background.replace_background_with_blur import (
    replace_background_with_blur,
)
from dreamstack.raster.effects.background.replace_background_with_gradient import (
    replace_background_with_gradient,
)

__all__: list[str] = [
    # Config classes
    "RemovalConfig",
    "MaskRefinementConfig",
    "GradientConfig",
    "ModelName",
    "GradientDirection",
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
