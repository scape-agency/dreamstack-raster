# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Add Operation
=============

Add two images together.

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


def add(
    image_a: NDArray[np.uint8],
    image_b: NDArray[np.uint8],
    *,
    factor: float = 1.0,
    clamp: bool = True,
) -> NDArray[np.uint8]:
    """Add two images together.

    Performs pixel-wise addition: result = A + (B * factor)

    Args:
        image_a: First image (base).
        image_b: Second image (to add).
        factor: Multiplier for second image (0-1).
        clamp: If True, clamp results to 0-255.

    Returns:
        Added image.

    Example:
        >>> combined = add(base, overlay)
        >>> # Add at 50% strength
        >>> combined = add(base, light, factor=0.5)
    """
    # Convert to float for computation
    a = image_a.astype(np.float32)
    b = image_b.astype(np.float32) * factor

    result = a + b

    if clamp:
        result = np.clip(result, 0, 255)

    return result.astype(np.uint8)
