# -*- coding: utf-8 -*-

"""Color space type enumeration."""

from __future__ import annotations

from enum import Enum


class ColorSpaceType(Enum):
    """ICC color space signatures."""

    XYZ = "XYZ "
    LAB = "Lab "
    LUV = "Luv "
    YCBCR = "YCbr"
    YXY = "Yxy "
    RGB = "RGB "
    GRAY = "GRAY"
    HSV = "HSV "
    HLS = "HLS "
    CMYK = "CMYK"
