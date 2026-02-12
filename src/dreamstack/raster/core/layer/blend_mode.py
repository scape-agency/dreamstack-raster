"""
Dreamstack Raster - Blend Mode
==============================

Layer blend modes enumeration.

"""

from enum import Enum, auto


class BlendMode(Enum):
    """
    Layer blend modes.

    These determine how a layer's pixels combine with layers below.
    """

    # Normal modes
    NORMAL = auto()
    DISSOLVE = auto()

    # Darken modes
    DARKEN = auto()
    MULTIPLY = auto()
    COLOR_BURN = auto()
    LINEAR_BURN = auto()
    DARKER_COLOR = auto()

    # Lighten modes
    LIGHTEN = auto()
    SCREEN = auto()
    COLOR_DODGE = auto()
    LINEAR_DODGE = auto()
    LIGHTER_COLOR = auto()

    # Contrast modes
    OVERLAY = auto()
    SOFT_LIGHT = auto()
    HARD_LIGHT = auto()
    VIVID_LIGHT = auto()
    LINEAR_LIGHT = auto()
    PIN_LIGHT = auto()
    HARD_MIX = auto()

    # Inversion modes
    DIFFERENCE = auto()
    EXCLUSION = auto()
    SUBTRACT = auto()
    DIVIDE = auto()

    # Component modes
    HUE = auto()
    SATURATION = auto()
    COLOR = auto()
    LUMINOSITY = auto()

    # Pass through (for groups)
    PASS_THROUGH = auto()
