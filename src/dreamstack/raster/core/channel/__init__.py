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
from dreamstack.raster.core.channel.channel_type import ChannelType

__all__ = [
    "ChannelType",
    "Channel",
    "ChannelManager",
]
