# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Blend Mode Compositing
======================

Layer blending with various blend modes.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from dreamstack.raster.compositing.blend.blend import blend
from dreamstack.raster.compositing.blend.blend_mode import BlendMode
from dreamstack.raster.compositing.blend.composite import composite
from dreamstack.raster.compositing.blend.merge_layers import (
    LayerInfo,
    merge_layers,
)

__all__: list[str] = [
    "BlendMode",
    "blend",
    "composite",
    "merge_layers",
    "LayerInfo",
]
