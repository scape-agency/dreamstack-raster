# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Minimum Operation
=================

Take minimum of two images per-pixel.

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


def minimum(
    image_a: NDArray[np.uint8],
    image_b: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    """Take minimum of two images per-pixel.

    For each pixel, takes the darker value from either image.

    Args:
        image_a: First image.
        image_b: Second image.

    Returns:
        Minimum image.
    """
    return np.minimum(image_a, image_b)
