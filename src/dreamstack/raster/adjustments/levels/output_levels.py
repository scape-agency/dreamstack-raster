# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Output levels adjustment function."""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from typing import TYPE_CHECKING

# =============================================================================
# Type Checking Imports
# =============================================================================

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.image import Image

from dreamstack.raster.adjustments.levels.levels import levels


def output_levels(
    image: Image, black_point: float = 0, white_point: float = 255
) -> Image:
    """
    Apply output levels only.

    Args:
        image: Input image
        black_point: Output black point
        white_point: Output white point

    Returns:
        Adjusted image
    """
    return levels(image, 0, 255, 1.0, black_point, white_point)
