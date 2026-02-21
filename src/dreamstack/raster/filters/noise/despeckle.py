# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Despeckle
=============================

Despeckle filter implementation.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dreamstack.raster.core.image import Image


def despeckle(image: Image) -> Image:
    """
    Apply despeckle filter (mild noise reduction).

    Args:
        image: Input image

    Returns:
        Despeckled image
    """
    from dreamstack.raster.filters.noise.median_filter import median_filter

    return median_filter(image, size=3)
