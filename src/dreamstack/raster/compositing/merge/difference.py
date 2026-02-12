"""
Difference Operation
====================

Calculate absolute difference between images.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def difference(
    image_a: NDArray[np.uint8],
    image_b: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    """Calculate absolute difference between images.

    Returns the absolute difference at each pixel.
    Useful for comparison and detecting changes.

    Args:
        image_a: First image.
        image_b: Second image.

    Returns:
        Difference image.

    Example:
        >>> changes = difference(before, after)
    """
    a = image_a.astype(np.int16)
    b = image_b.astype(np.int16)

    result = np.abs(a - b)

    return result.astype(np.uint8)
