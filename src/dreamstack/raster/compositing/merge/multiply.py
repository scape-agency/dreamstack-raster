# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Multiply Operation
==================

Multiply two images.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def multiply(
    image_a: NDArray[np.uint8],
    image_b: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    """Multiply two images.

    Performs pixel-wise multiplication (normalized).
    Result is always darker or equal to inputs.

    Args:
        image_a: First image.
        image_b: Second image.

    Returns:
        Multiplied image.

    Example:
        >>> darkened = multiply(image, shadow_mask)
    """
    a = image_a.astype(np.float32) / 255.0
    b = image_b.astype(np.float32) / 255.0

    result = a * b * 255.0

    return np.clip(result, 0, 255).astype(np.uint8)
