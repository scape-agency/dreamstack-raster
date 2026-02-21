# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Input levels adjustment function."""


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


def input_levels(
    image: Image,
    black_point: float = 0,
    white_point: float = 255,
    gamma: float = 1.0,
) -> Image:
    """
    Apply input levels only.

    Args:
        image: Input image
        black_point: Input black point
        white_point: Input white point
        gamma: Midtone gamma

    Returns:
        Adjusted image
    """
    return levels(image, black_point, white_point, gamma, 0, 255)
