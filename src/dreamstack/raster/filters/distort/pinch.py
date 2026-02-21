# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Pinch Distortion
====================================

Pinch distortion filter implementation.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.image import Image


def pinch(
    image: Image,
    amount: float = 50,
    center: tuple[float, float] | None = None,
) -> Image:
    """
    Apply pinch distortion.

    Args:
        image: Input image
        amount: Pinch amount (0-100)
        center: Effect center (relative 0-1)

    Returns:
        Distorted image
    """
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.filters.distort.sphere import sphere

    return sphere(image, -amount, center)
