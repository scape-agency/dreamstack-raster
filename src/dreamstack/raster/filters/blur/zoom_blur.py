# -*- coding: utf-8 -*-

"""
Dreamstack Raster - Zoom Blur
=============================

Zoom blur filter implementation.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from dreamstack.raster.core.image import Image


def zoom_blur(
    image: Image,
    amount: float = 10,
    center: Optional[Tuple[float, float]] = None,
) -> Image:
    """
    Apply zoom blur (radial blur in zoom mode).

    Args:
        image: Input image
        amount: Blur amount
        center: Blur center (relative 0-1)

    Returns:
        Blurred image
    """
    from dreamstack.raster.filters.blur.radial_blur import radial_blur

    return radial_blur(image, amount, center, mode="zoom")
