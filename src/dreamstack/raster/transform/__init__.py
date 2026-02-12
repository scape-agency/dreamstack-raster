"""
Dreamstack Raster - Transform Module
====================================

Image transformation operations: resize, rotate, crop,
perspective, and geometric warping.

"""

from __future__ import annotations

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
from dreamstack.raster.transform.rotate import (
    arbitrary_rotate,
    flip_both,
    flip_horizontal,
    flip_vertical,
    get_rotation_matrix,
    random_rotate,
    rotate,
    rotate_90,
    rotate_180,
    rotate_270,
    rotate_point,
    rotate_points,
)
from dreamstack.raster.transform.translate import (
    apply_affine_matrix,
    center_to_origin,
    get_translation_matrix,
    random_translate,
    translate,
    translate_percentage,
)
from dreamstack.raster.transform.upscale import (
    ImageUpscaler,
    UpscaleConfig,
    upscale_image,
    upscale_lanczos,
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
    "ImageUpscaler",
    "UpscaleConfig",
    "upscale_lanczos",
    "upscale_image",
    # Rotate
    "rotate",
    "rotate_90",
    "rotate_180",
    "rotate_270",
    "flip_horizontal",
    "flip_vertical",
    "flip_both",
    "arbitrary_rotate",
    "random_rotate",
    "get_rotation_matrix",
    "rotate_point",
    "rotate_points",
    # Translate
    "translate",
    "translate_percentage",
    "random_translate",
    "center_to_origin",
    "get_translation_matrix",
    "apply_affine_matrix",
]
