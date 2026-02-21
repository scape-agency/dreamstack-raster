# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""S-curve adjustment function."""


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

from dreamstack.raster.adjustments.curves.create_curve import create_curve
from dreamstack.raster.adjustments.curves.curves import curves


def s_curve(image: Image, strength: float = 50) -> Image:
    """
    Apply S-curve for contrast enhancement.

    Args:
        image: Input image
        strength: Curve strength (0-100)

    Returns:
        Adjusted image
    """
    # Create S-curve
    mid_offset = strength / 2

    curve = create_curve(
        [
            (0, 0),
            (64, 64 - mid_offset),
            (128, 128),
            (192, 192 + mid_offset),
            (255, 255),
        ]
    )

    return curves(image, rgb_curve=curve)
