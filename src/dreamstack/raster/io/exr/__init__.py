# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - OpenEXR Support
===================================

High dynamic range EXR file format support.

"""

from dreamstack.raster.io.exr.get_exr_channels import get_exr_channels
from dreamstack.raster.io.exr.get_exr_info import get_exr_info
from dreamstack.raster.io.exr.load_exr import load_exr
from dreamstack.raster.io.exr.save_exr import save_exr

__all__: list[str] = [
    "load_exr",
    "save_exr",
    "get_exr_channels",
    "get_exr_info",
]
