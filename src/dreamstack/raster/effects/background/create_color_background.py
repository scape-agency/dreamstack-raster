# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Create Color Background
=======================

Create solid color background images.

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


def create_color_background(
    size: tuple[int, int],
    color: tuple[int, int, int] = (255, 255, 255),
) -> NDArray[np.uint8]:
    """Create a solid color background image.

    Args:
        size: (width, height) of the background.
        color: RGB color tuple.

    Returns:
        RGB background image.
    """
    height, width = size[1], size[0]
    bg = np.zeros((height, width, 3), dtype=np.uint8)
    bg[:, :] = color
    return bg
