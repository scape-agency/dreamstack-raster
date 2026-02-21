# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Subtract Operation
==================

Subtract one image from another.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

# =============================================================================
# Type Checking Imports
# =============================================================================

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel
    from numpy.typing import NDArray


def subtract(
    image_a: NDArray[np.uint8],
    image_b: NDArray[np.uint8],
    *,
    factor: float = 1.0,
    clamp: bool = True,
) -> NDArray[np.uint8]:
    """Subtract one image from another.

    Performs pixel-wise subtraction: result = A - (B * factor)

    Args:
        image_a: Base image.
        image_b: Image to subtract.
        factor: Multiplier for subtracted image.
        clamp: If True, clamp results to 0-255.

    Returns:
        Subtracted image.

    Example:
        >>> diff = subtract(image_a, image_b)
    """
    a = image_a.astype(np.float32)
    b = image_b.astype(np.float32) * factor

    result = a - b

    if clamp:
        result = np.clip(result, 0, 255)

    return result.astype(np.uint8)
