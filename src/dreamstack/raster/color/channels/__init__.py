# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Color Channel Operations
========================

Operations for splitting, extracting, and merging color channels
from images. Essential for machine learning preprocessing and
image analysis tasks.
"""

from .channel_to_grayscale_rgb import channel_to_grayscale_rgb
from .extract_channel import ChannelName, extract_channel
from .extract_rgb_arrays import extract_rgb_arrays
from .isolate_channel import isolate_channel
from .merge_channels import merge_channels
from .split_channels import split_channels
from .swap_channels import swap_channels

__all__ = [
    "ChannelName",
    "channel_to_grayscale_rgb",
    "extract_channel",
    "extract_rgb_arrays",
    "isolate_channel",
    "merge_channels",
    "split_channels",
    "swap_channels",
]
