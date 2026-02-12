"""
Mask Operations
===============

Mask creation and application for compositing.

"""

from __future__ import annotations

from dreamstack.raster.compositing.mask.apply_mask import apply_mask
from dreamstack.raster.compositing.mask.channel_mask import channel_mask
from dreamstack.raster.compositing.mask.clipping_mask import (
    create_clipping_mask,
)
from dreamstack.raster.compositing.mask.luminosity_mask import luminosity_mask

__all__: list[str] = [
    "apply_mask",
    "create_clipping_mask",
    "luminosity_mask",
    "channel_mask",
]
