# -*- coding: utf-8 -*-

"""
Dreamstack Raster - Bulge Distortion
====================================

Bulge distortion filter implementation.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from dreamstack.raster.core.image import Image


def bulge(
    image: Image,
    amount: float = 50,
    center: Optional[Tuple[float, float]] = None,
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
