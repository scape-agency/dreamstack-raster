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
from dreamstack.raster.compositing.layers import (
    apply_alpha_from_mask,
    composite_with_mask,
    generate_layer_combinations,
    generate_layer_stack_from_dirs,
    stack_layers,
)
from dreamstack.raster.compositing.mask import (
    apply_mask,
    channel_mask,
    create_clipping_mask,
    luminosity_mask,
)
from dreamstack.raster.compositing.merge import (
    MergeMode,
    add,
    average,
    difference,
    divide,
    maximum,
    merge,
    minimum,
    multiply,
    over,
    screen,
    subtract,
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
    # Layers
    "stack_layers",
    "generate_layer_combinations",
    "generate_layer_stack_from_dirs",
    "composite_with_mask",
    "apply_alpha_from_mask",
    # Merge Operations
    "MergeMode",
    "add",
    "subtract",
    "multiply",
    "divide",
    "screen",
    "difference",
    "average",
    "maximum",
    "minimum",
    "merge",
    "over",
]
