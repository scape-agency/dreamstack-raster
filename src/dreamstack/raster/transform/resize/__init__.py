"""
Dreamstack Raster - Resize Module
=================================

Image resizing, scaling, and dimension manipulation utilities.

"""

# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from .downscale import downscale
from .fill import fill
from .fit import fit
from .fit_to_dimensions import fit_to_dimensions
from .pad_to_aspect import pad_to_aspect
from .resize import resize
from .resize_config import ResizeConfig
from .resize_for_ai import resize_for_ai
from .resize_method import ResizeMethod
from .resize_to_aspect import resize_to_aspect
from .resize_to_width import resize_to_width
from .scale import scale
from .thumbnail import thumbnail
from .upscale import upscale

__all__: list[str] = [
    "ResizeMethod",
    "ResizeConfig",
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
