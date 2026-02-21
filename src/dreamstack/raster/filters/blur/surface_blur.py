# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Surface Blur
================================

Surface blur (edge-preserving) filter implementation.

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


def surface_blur(
    image: Image, radius: int = 10, threshold: float = 15
) -> Image:
    """
    Apply surface blur (preserves edges while smoothing).

    Args:
        image: Input image
        radius: Blur radius
        threshold: Edge threshold

    Returns:
        Blurred image
    """
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.filters.blur.bilateral_blur import bilateral_blur

    return bilateral_blur(image, radius, threshold, threshold * 3)
