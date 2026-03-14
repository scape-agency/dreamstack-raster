# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Black & White Adjustments
=============================================

Black and white conversion, channel mixing, and gradient mapping.

"""

from dreamstack.raster.adjustments.black_white.black_white import black_white
from dreamstack.raster.adjustments.black_white.channel_mixer import (
    channel_mixer,
)
from dreamstack.raster.adjustments.black_white.desaturate import desaturate
from dreamstack.raster.adjustments.black_white.duotone import duotone
from dreamstack.raster.adjustments.black_white.gradient_map import gradient_map
from dreamstack.raster.adjustments.black_white.invert import invert
from dreamstack.raster.adjustments.black_white.sepia import sepia
from dreamstack.raster.adjustments.black_white.threshold import threshold
from dreamstack.raster.adjustments.black_white.tritone import tritone

__all__: list[str] = [
    "black_white",
    "desaturate",
    "channel_mixer",
    "gradient_map",
    "duotone",
    "tritone",
    "sepia",
    "invert",
    "threshold",
]
