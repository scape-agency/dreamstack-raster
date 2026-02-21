# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Channel Count
=================================

Channel count mapping for pixel formats.

"""

from dreamstack.raster.core.pixel.pixel_format import PixelFormat

CHANNEL_COUNT = {
    PixelFormat.GRAY: 1,
    PixelFormat.GRAY_ALPHA: 2,
    PixelFormat.RGB: 3,
    PixelFormat.RGBA: 4,
    PixelFormat.CMYK: 4,
    PixelFormat.LAB: 3,
    PixelFormat.HSV: 3,
    PixelFormat.HSL: 3,
}
