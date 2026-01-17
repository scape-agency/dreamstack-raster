# -*- coding: utf-8 -*-

"""
Dreamstack Raster - Transform Module
====================================

Image transformation operations: resize, rotate, crop,
perspective, and geometric warping.

"""

from __future__ import annotations

from dreamstack.raster.transform.crop import (
    auto_crop,
    canvas_size,
    content_aware_crop,
    crop,
    trim,
)
from dreamstack.raster.transform.perspective import (
    distort,
    perspective,
    perspective_crop,
    skew,
    warp,
)
from dreamstack.raster.transform.resize import (
    ResizeMethod,
    fill,
    fit,
    resize,
    scale,
    thumbnail,
)
from dreamstack.raster.transform.rotate import (
    arbitrary_rotate,
    flip_horizontal,
    flip_vertical,
    rotate,
    rotate_90,
    rotate_180,
    rotate_270,
)

__all__: list[str] = [
    # Resize
    "resize",
    "scale",
    "fit",
    "fill",
    "thumbnail",
    "ResizeMethod",
    # Rotate
    "rotate",
    "rotate_90",
    "rotate_180",
    "rotate_270",
    "flip_horizontal",
    "flip_vertical",
    "arbitrary_rotate",
    # Crop
    "crop",
    "canvas_size",
    "trim",
    "auto_crop",
    "content_aware_crop",
    # Perspective
    "perspective",
    "perspective_crop",
    "skew",
    "distort",
    "warp",
]
