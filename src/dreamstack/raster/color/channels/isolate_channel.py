# -*- coding: utf-8 -*-

"""Isolate single channel operation."""

from __future__ import annotations

from typing import Literal, Union

import numpy as np
from numpy.typing import NDArray

ChannelName = Literal["red", "green", "blue", "alpha", "r", "g", "b", "a"]


def isolate_channel(
    image: NDArray[np.uint8],
    channel: Union[int, ChannelName],
    color_format: str = "BGR",
) -> NDArray[np.uint8]:
    """Create image with only one channel visible (others zeroed).

    Useful for visualizing individual color contributions.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input color image (H, W, C).
    channel : int or str
        Channel to keep visible.
    color_format : str, optional
        Color format: "BGR" or "RGB". Default is "BGR".

    Returns
    -------
    NDArray[np.uint8]
        Color image with only specified channel (H, W, C).

    Examples
    --------
    >>> red_only = isolate_channel(img, "red")
    >>> # Will show image with only red channel, green and blue are black
    """
    result = np.zeros_like(image)

    if isinstance(channel, str):
        channel = channel.lower()
        channel_map_bgr = {
            "blue": 0,
            "b": 0,
            "green": 1,
            "g": 1,
            "red": 2,
            "r": 2,
        }
        channel_map_rgb = {
            "red": 0,
            "r": 0,
            "green": 1,
            "g": 1,
            "blue": 2,
            "b": 2,
        }
        channel_map = (
            channel_map_bgr
            if color_format.upper() in ("BGR", "BGRA")
            else channel_map_rgb
        )
        channel_idx = channel_map.get(channel, 0)
    else:
        channel_idx = channel

    result[:, :, channel_idx] = image[:, :, channel_idx]
    return result
