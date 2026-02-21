# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Layer combination generation."""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

import itertools
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING

import numpy as np

from .stack_layers import stack_layers

if TYPE_CHECKING:
    from numpy.typing import NDArray


def generate_layer_combinations(
    layer_variants: Sequence[Sequence[NDArray[np.uint8]]],
) -> Iterator[NDArray[np.uint8]]:
    """Generate all combinations of layer variants.

    Given multiple layers each with multiple variants, generates all
    possible combinations by stacking one variant from each layer.

    Args:
        layer_variants: List of layers, each containing list of variant images.

    Yields:
        Composited images for each combination.

    Example:
        >>> # 3 variants each for 3 layers = 27 combinations
        >>> variants = [
        ...     [bg1, bg2, bg3],        # Background variants
        ...     [mid1, mid2, mid3],     # Midground variants
        ...     [fg1, fg2, fg3],        # Foreground variants
        ... ]
        >>> for combo in generate_layer_combinations(variants):
        ...     save_image(combo, f"combo_{i}.png")
    """
    for combo in itertools.product(*layer_variants):
        yield stack_layers(list(combo))
