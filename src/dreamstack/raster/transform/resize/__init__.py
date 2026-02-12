# -*- coding: utf-8 -*-

"""
Dreamstack Raster - Resize Module
=================================

Image resizing, scaling, and dimension manipulation utilities.

"""

from __future__ import annotations

from dreamstack.raster.transform.resize.operations import (
    ResizeMethod,
    downscale,
    fill,
    fit,
    fit_to_dimensions,
    pad_to_aspect,
    resize,
    resize_for_ai,
    resize_to_aspect,
    resize_to_width,
    scale,
    thumbnail,
    upscale,
)

__all__: list[str] = [
    "ResizeMethod",
    "resize",
    "scale",
    "fit",
    "fill",
    "resize_to_width",
    "resize_to_aspect",
    "resize_for_ai",
    "fit_to_dimensions",
    "thumbnail",
    "downscale",
    "upscale",
    "pad_to_aspect",
]
