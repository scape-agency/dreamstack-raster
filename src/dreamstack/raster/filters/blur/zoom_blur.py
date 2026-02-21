# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Zoom Blur
=============================

Zoom blur filter implementation.

"""


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


def zoom_blur(
    image: Image,
    amount: float = 10,
    center: tuple[float, float] | None = None,
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
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.filters.blur.radial_blur import radial_blur

    return radial_blur(image, amount, center, mode="zoom")
