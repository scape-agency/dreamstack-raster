# -*- coding: utf-8 -*-

"""
Dreamstack Raster - Compositing Module
======================================

Layer compositing: blend modes, alpha blending, clipping masks.

"""

from __future__ import annotations

from dreamstack.raster.compositing.alpha import (
    alpha_composite,
    extract_alpha,
    premultiply_alpha,
    set_alpha,
    unpremultiply_alpha,
)
from dreamstack.raster.compositing.blend import (
    BlendMode,
    blend,
    composite,
    merge_layers,
)
from dreamstack.raster.compositing.mask import (
    apply_mask,
    channel_mask,
    create_clipping_mask,
    luminosity_mask,
)

__all__: list[str] = [
    # Blend
    "BlendMode",
    "blend",
    "composite",
    "merge_layers",
    # Alpha
    "alpha_composite",
    "premultiply_alpha",
    "unpremultiply_alpha",
    "set_alpha",
    "extract_alpha",
    # Mask
    "apply_mask",
    "create_clipping_mask",
    "luminosity_mask",
    "channel_mask",
]
