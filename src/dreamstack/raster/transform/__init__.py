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
)
from dreamstack.raster.transform.upscale import (
    BaseUpscaler,
    ImageUpscaler,
    UpscaleConfig,
    upscale_2x,
    upscale_4x,
    upscale_image,
    upscale_lanczos,
    upscale_to_size,
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
    "resize_to_width",
    "resize_to_aspect",
    "resize_for_ai",
    "fit_to_dimensions",
    "downscale",
    "pad_to_aspect",
    # Upscale
    "BaseUpscaler",
    "ImageUpscaler",
    "UpscaleConfig",
    "upscale_lanczos",
    "upscale_image",
    "upscale_to_size",
    "upscale_2x",
    "upscale_4x",
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
