# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Channel Management
======================================

Individual channel handling for advanced image manipulation.

"""

from dreamstack.raster.core.channel.channel import Channel
from dreamstack.raster.core.channel.channel_manager import ChannelManager
from dreamstack.raster.core.channel.channel_ops import (
    ChannelName,
    channel_to_grayscale_rgb,
    extract_channel,
    extract_rgb_arrays,
    isolate_channel,
    merge_channels,
    split_channels,
    swap_channels,
)
from dreamstack.raster.core.channel.channel_type import ChannelType

__all__: list[str] = [
    "ChannelType",
    "Channel",
    "ChannelManager",
    "ChannelName",
    "split_channels",
    "merge_channels",
    "extract_channel",
    "isolate_channel",
    "swap_channels",
    "channel_to_grayscale_rgb",
    "extract_rgb_arrays",
]
