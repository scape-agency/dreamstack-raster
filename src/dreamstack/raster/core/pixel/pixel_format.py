# -*- coding: utf-8 -*-

"""
Dreamstack Raster - Pixel Format
================================

Supported pixel format enumeration.

"""

from enum import Enum, auto


class PixelFormat(Enum):
    """Supported pixel formats."""

    GRAY = auto()  # Single channel grayscale
    GRAY_ALPHA = auto()  # Grayscale with alpha
    RGB = auto()  # 3 channel RGB
    RGBA = auto()  # 4 channel RGBA
    CMYK = auto()  # 4 channel CMYK
    LAB = auto()  # 3 channel CIE Lab
    HSV = auto()  # 3 channel HSV
    HSL = auto()  # 3 channel HSL
