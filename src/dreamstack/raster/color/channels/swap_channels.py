# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""Swap channels operation."""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

ChannelName = Literal["red", "green", "blue", "alpha", "r", "g", "b", "a"]


def swap_channels(
    image: NDArray[np.uint8],
    # pylint: disable=import-outside-toplevel
    from_channel: int | ChannelName,
    to_channel: int | ChannelName,
    color_format: str = "BGR",
) -> NDArray[np.uint8]:
    """Swap two color channels.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input color image.
    # pylint: disable=import-outside-toplevel
    from_channel : int or str
        First channel to swap.
    to_channel : int or str
        Second channel to swap.
    color_format : str, optional
        Color format. Default is "BGR".

    Returns
    -------
    NDArray[np.uint8]
        Image with channels swapped.
    """
    channel_map_bgr = {"blue": 0, "b": 0, "green": 1, "g": 1, "red": 2, "r": 2}
    channel_map_rgb = {"red": 0, "r": 0, "green": 1, "g": 1, "blue": 2, "b": 2}
    channel_map = (
        channel_map_bgr
        if color_format.upper() in ("BGR", "BGRA")
        else channel_map_rgb
    )

    def get_idx(ch):
        if isinstance(ch, str):
            return channel_map.get(ch.lower(), 0)
        return ch

    idx1 = get_idx(from_channel)
    idx2 = get_idx(to_channel)

    result = image.copy()
    result[:, :, idx1] = image[:, :, idx2]
    result[:, :, idx2] = image[:, :, idx1]
    return result
