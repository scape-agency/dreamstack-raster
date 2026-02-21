# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""Duotone effect function."""


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

from dreamstack.raster.adjustments.black_white.gradient_map import gradient_map


def duotone(
    image: Image,
    shadow_color: tuple[int, int, int] = (0, 0, 0),
    highlight_color: tuple[int, int, int] = (255, 255, 255),
) -> Image:
    """
    Apply duotone effect (two-color gradient map).

    Args:
        image: Input image
        shadow_color: Shadow color (R, G, B) 0-255
        highlight_color: Highlight color (R, G, B) 0-255

    Returns:
        Duotone image
    """
    gradient: list[tuple[float, tuple[int, int, int]]] = [
        (0.0, shadow_color),
        (1.0, highlight_color),
    ]
    return gradient_map(image, gradient)
