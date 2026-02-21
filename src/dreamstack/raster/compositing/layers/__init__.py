# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Layer Stacking Operations
=========================

Multi-layer image compositing and combination utilities.

"""

from .apply_alpha_from_mask import apply_alpha_from_mask
from .composite_with_mask import composite_with_mask
from .generate_layer_combinations import generate_layer_combinations
from .generate_layer_stack_from_dirs import generate_layer_stack_from_dirs
from .layer_stack_config import LayerStackConfig
from .stack_layers import stack_layers

__all__ = [
    "LayerStackConfig",
    "stack_layers",
    "generate_layer_combinations",
    "generate_layer_stack_from_dirs",
    "composite_with_mask",
    "apply_alpha_from_mask",
]
