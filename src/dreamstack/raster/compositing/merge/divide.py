# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Divide Operation
================

Divide one image by another.

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


def divide(
    image_a: NDArray[np.uint8],
    image_b: NDArray[np.uint8],
    *,
    epsilon: float = 1e-6,
) -> NDArray[np.uint8]:
    """Divide one image by another.

    Performs pixel-wise division. Used for color correction
    and removing lighting variations.

    Args:
        image_a: Numerator image.
        image_b: Denominator image.
        epsilon: Small value to prevent division by zero.

    Returns:
        Divided image.

    Example:
        >>> corrected = divide(image, light_pattern)
    """
    a = image_a.astype(np.float32)
    b = image_b.astype(np.float32) + epsilon

    result = (a / b) * 128.0  # Normalize to mid-gray

    return np.clip(result, 0, 255).astype(np.uint8)
