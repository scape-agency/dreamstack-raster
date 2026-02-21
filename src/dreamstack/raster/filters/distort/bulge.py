# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Bulge Distortion
====================================

Bulge distortion filter implementation.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dreamstack.raster.core.image import Image


def bulge(
    image: Image,
    amount: float = 50,
    center: tuple[float, float] | None = None,
) -> Image:
    """
    Apply bulge distortion.

    Args:
        image: Input image
        amount: Bulge amount (0-100)
        center: Effect center (relative 0-1)

    Returns:
        Distorted image
    """
    from dreamstack.raster.filters.distort.sphere import sphere

    return sphere(image, amount, center)
