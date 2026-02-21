# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Average Operation
=================

Average two images.

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


def average(
    image_a: NDArray[np.uint8],
    image_b: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    """Average two images.

    Simple 50/50 blend of two images.

    Args:
        image_a: First image.
        image_b: Second image.

    Returns:
        Averaged image.
    """
    a = image_a.astype(np.float32)
    b = image_b.astype(np.float32)

    result = (a + b) / 2.0

    return result.astype(np.uint8)
