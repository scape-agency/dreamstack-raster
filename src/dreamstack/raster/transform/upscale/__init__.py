# -*- coding: utf-8 -*-

"""
Dreamstack Raster - Upscale Module
==================================

AI-based image upscaling using PyTorch models.
Supports various super-resolution models.

"""

from __future__ import annotations

from dreamstack.raster.transform.upscale.upscaler import (
    ImageUpscaler,
    UpscaleConfig,
)
from dreamstack.raster.transform.upscale.operations import (
    upscale_image,
    upscale_lanczos,
)

__all__: list[str] = [
    "ImageUpscaler",
    "UpscaleConfig",
    "upscale_image",
    "upscale_lanczos",
]
