# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Screen Operation
================

Screen blend two images.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel
    from numpy.typing import NDArray


def screen(
    image_a: NDArray[np.uint8],
    image_b: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    """Screen blend two images.

    Inverse of multiply. Result is always lighter or equal.
    Formula: 1 - (1-A) * (1-B)

    Args:
        image_a: First image.
        image_b: Second image.

    Returns:
        Screen blended image.

    Example:
        >>> lightened = screen(image, light_effect)
    """
    a = image_a.astype(np.float32) / 255.0
    b = image_b.astype(np.float32) / 255.0

    result = 1.0 - (1.0 - a) * (1.0 - b)

    return (result * 255.0).astype(np.uint8)
