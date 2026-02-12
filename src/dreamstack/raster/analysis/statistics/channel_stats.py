"""
ChannelStats Dataclass
======================

Statistics for a single channel.

"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ChannelStats:
    """Statistics for a single channel.

    Attributes
    ----------
    mean : float
        Mean value.
    std : float
        Standard deviation.
    min : int
        Minimum value.
    max : int
        Maximum value.
    median : float
        Median value.
    """

    mean: float
    std: float
    min: int
    max: int
    median: float
