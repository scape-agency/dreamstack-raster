# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Preset curves function."""


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


def preset_curves(image: Image, preset: str) -> Image:
    """
    Apply preset curves.

    Args:
        image: Input image
        preset: Preset name ('contrast', 'lighten', 'darken', 'fade',
                            'cross_process', 'vintage', 'matte')

    Returns:
        Adjusted image
    """
    presets = {
        "contrast": create_curve([(0, 0), (64, 48), (192, 208), (255, 255)]),
        "lighten": create_curve([(0, 0), (128, 160), (255, 255)]),
        "darken": create_curve([(0, 0), (128, 96), (255, 255)]),
        "fade": create_curve([(0, 32), (255, 224)]),
        "cross_process": None,  # Complex - per channel
        "vintage": create_curve([(0, 16), (32, 32), (224, 224), (255, 240)]),
        "matte": create_curve([(0, 24), (64, 80), (192, 200), (255, 232)]),
    }

    if preset == "cross_process":
        # Cross-process look with per-channel curves
        red = create_curve([(0, 0), (128, 148), (255, 255)])
        green = create_curve([(0, 0), (128, 128), (255, 255)])
        blue = create_curve([(0, 24), (128, 108), (255, 232)])
        return curves(image, red_curve=red, green_curve=green, blue_curve=blue)

    curve = presets.get(preset)
    if curve:
        return curves(image, rgb_curve=curve)

    return image.copy()
