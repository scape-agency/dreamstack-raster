"""
ImageStats Dataclass
====================

Statistics for an entire image.

"""

from __future__ import annotations

from dataclasses import dataclass

from .channel_stats import ChannelStats


@dataclass
class ImageStats:
    """Statistics for an entire image.

    Attributes
    ----------
    width : int
        Image width.
    height : int
        Image height.
    channels : int
        Number of channels.
    dtype : str
        Data type.
    channel_stats : list
        Per-channel statistics.
    """

    width: int
    height: int
    channels: int
    dtype: str
    channel_stats: list[ChannelStats]
