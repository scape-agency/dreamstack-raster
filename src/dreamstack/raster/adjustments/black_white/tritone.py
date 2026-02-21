# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Tritone effect function."""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dreamstack.raster.core.image import Image

from dreamstack.raster.adjustments.black_white.gradient_map import gradient_map


def tritone(
    image: Image,
    shadow_color: tuple[int, int, int] = (0, 0, 0),
    midtone_color: tuple[int, int, int] = (128, 128, 128),
    highlight_color: tuple[int, int, int] = (255, 255, 255),
) -> Image:
    """
    Apply tritone effect (three-color gradient map).

    Args:
        image: Input image
        shadow_color: Shadow color (R, G, B) 0-255
        midtone_color: Midtone color (R, G, B) 0-255
        highlight_color: Highlight color (R, G, B) 0-255

    Returns:
        Tritone image
    """
    gradient = [(0, shadow_color), (0.5, midtone_color), (1, highlight_color)]
    return gradient_map(image, gradient)
