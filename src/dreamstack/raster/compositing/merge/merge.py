# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Merge Operation
===============

Merge two images using specified mode.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from dreamstack.raster.compositing.merge.add import add
from dreamstack.raster.compositing.merge.difference import difference
from dreamstack.raster.compositing.merge.divide import divide
from dreamstack.raster.compositing.merge.multiply import multiply
from dreamstack.raster.compositing.merge.screen import screen
from dreamstack.raster.compositing.merge.subtract import subtract

MergeMode = Literal[
    "add", "subtract", "multiply", "divide", "screen", "difference"
]


def merge(
    image_a: NDArray[np.uint8],
    image_b: NDArray[np.uint8],
    mode: MergeMode = "add",
    *,
    factor: float = 1.0,
) -> NDArray[np.uint8]:
    """Merge two images using specified mode.

    Convenience function for applying different merge operations.

    Args:
        image_a: First image.
        image_b: Second image.
        mode: Merge mode (add, subtract, multiply, divide, screen, difference).
        factor: Factor for add/subtract modes.

    Returns:
        Merged image.

    Example:
        >>> result = merge(a, b, mode="multiply")
    """
    operations = {
        "add": lambda: add(image_a, image_b, factor=factor),
        "subtract": lambda: subtract(image_a, image_b, factor=factor),
        "multiply": lambda: multiply(image_a, image_b),
        "divide": lambda: divide(image_a, image_b),
        "screen": lambda: screen(image_a, image_b),
        "difference": lambda: difference(image_a, image_b),
    }

    if mode not in operations:
        raise ValueError(f"Unknown merge mode: {mode}")

    return operations[mode]()
