# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Image Statistics Module
=======================

Statistical analysis functions for images.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from .channel_statistics import channel_statistics
from .channel_stats import ChannelStats
from .color_count import color_count
from .image_statistics import image_statistics
from .image_stats import ImageStats
from .unique_colors import unique_colors

__all__ = [
    "ChannelStats",
    "ImageStats",
    "channel_statistics",
    "image_statistics",
    "color_count",
    "unique_colors",
]
