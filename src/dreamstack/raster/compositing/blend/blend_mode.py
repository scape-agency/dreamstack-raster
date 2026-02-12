"""
Blend Mode Enum
===============

Defines available blend modes for layer compositing.

"""

from __future__ import annotations

from enum import Enum, auto


class BlendMode(Enum):
    """Blend modes for layer compositing.

    These modes define how two layers are combined visually.

    Basic Modes:
        NORMAL: Standard alpha blending.
        DISSOLVE: Random pixel selection based on opacity.

    Darken Modes:
        DARKEN: Keep darker pixels.
        MULTIPLY: Multiply colors (always darkens).
        COLOR_BURN: Increase contrast, darken.
        LINEAR_BURN: Decrease brightness.
        DARKER_COLOR: Keep darker color.

    Lighten Modes:
        LIGHTEN: Keep lighter pixels.
        SCREEN: Inverse multiply (always lightens).
        COLOR_DODGE: Decrease contrast, brighten.
        LINEAR_DODGE: Add brightness.
        LIGHTER_COLOR: Keep lighter color.

    Contrast Modes:
        OVERLAY: Multiply or screen based on base.
        SOFT_LIGHT: Gentle contrast adjustment.
        HARD_LIGHT: Strong contrast adjustment.
        VIVID_LIGHT: Color burn or dodge.
        LINEAR_LIGHT: Linear burn or dodge.
        PIN_LIGHT: Replace lighter/darker pixels.
        HARD_MIX: Threshold posterization.

    Comparative Modes:
        DIFFERENCE: Absolute difference.
        EXCLUSION: Softer difference.
        SUBTRACT: Subtract colors.
        DIVIDE: Divide colors.

    Component Modes:
        HUE: Blend hue only.
        SATURATION: Blend saturation only.
        COLOR: Blend hue and saturation.
        LUMINOSITY: Blend luminosity only.
    """

    # Basic
    NORMAL = auto()
    DISSOLVE = auto()

    # Darken
    DARKEN = auto()
    MULTIPLY = auto()
    COLOR_BURN = auto()
    LINEAR_BURN = auto()
    DARKER_COLOR = auto()

    # Lighten
    LIGHTEN = auto()
    SCREEN = auto()
    COLOR_DODGE = auto()
    LINEAR_DODGE = auto()
    LIGHTER_COLOR = auto()

    # Contrast
    OVERLAY = auto()
    SOFT_LIGHT = auto()
    HARD_LIGHT = auto()
    VIVID_LIGHT = auto()
    LINEAR_LIGHT = auto()
    PIN_LIGHT = auto()
    HARD_MIX = auto()

    # Comparative
    DIFFERENCE = auto()
    EXCLUSION = auto()
    SUBTRACT = auto()
    DIVIDE = auto()

    # Component (HSL)
    HUE = auto()
    SATURATION = auto()
    COLOR = auto()
    LUMINOSITY = auto()
