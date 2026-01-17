# -*- coding: utf-8 -*-

"""
Dreamstack Raster - Channel Type
================================

Types of image channels enumeration.

"""

from enum import Enum, auto


class ChannelType(Enum):
    """Types of image channels."""

    # Color channels
    RED = auto()
    GREEN = auto()
    BLUE = auto()
    ALPHA = auto()

    # CMYK
    CYAN = auto()
    MAGENTA = auto()
    YELLOW = auto()
    BLACK = auto()

    # LAB
    LIGHTNESS = auto()
    A_CHANNEL = auto()  # Green-Red axis
    B_CHANNEL = auto()  # Blue-Yellow axis

    # HSV/HSL
    HUE = auto()
    SATURATION = auto()
    VALUE = auto()
    LUMINOSITY = auto()

    # Grayscale
    GRAY = auto()

    # Mask channels
    MASK = auto()
    SPOT = auto()

    # Custom
    CUSTOM = auto()
