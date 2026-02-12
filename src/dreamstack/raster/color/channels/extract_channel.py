# -*- coding: utf-8 -*-

"""Extract single channel operation."""

from __future__ import annotations

from typing import Literal, Union

import numpy as np
from numpy.typing import NDArray

ChannelName = Literal["red", "green", "blue", "alpha", "r", "g", "b", "a"]


def extract_channel(
    image: NDArray[np.uint8],
    channel: Union[int, ChannelName],
    color_format: str = "BGR",
) -> NDArray[np.uint8]:
    """Extract a single channel from an image.

    Retrieves one specific color channel from a multi-channel image.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input color image (H, W, C).
    channel : int or str
        Channel to extract. Can be:
        - Integer index (0, 1, 2, 3)
        - Channel name: "red", "green", "blue", "alpha" or "r", "g", "b", "a"
    color_format : str, optional
        Color format: "BGR" or "RGB". Default is "BGR".

    Returns
    -------
    NDArray[np.uint8]
        Single channel image (H, W).

    Examples
    --------
    >>> red = extract_channel(img, "red")
    >>> blue = extract_channel(img, 2)  # By index
    """
    if isinstance(channel, str):
        channel = channel.lower()
        channel_map_bgr = {
            "blue": 0,
            "b": 0,
            "green": 1,
            "g": 1,
            "red": 2,
            "r": 2,
            "alpha": 3,
            "a": 3,
        }
        channel_map_rgb = {
            "red": 0,
            "r": 0,
            "green": 1,
            "g": 1,
            "blue": 2,
            "b": 2,
            "alpha": 3,
            "a": 3,
        }
        channel_map = (
            channel_map_bgr
            if color_format.upper() in ("BGR", "BGRA")
            else channel_map_rgb
        )
        if channel not in channel_map:
            raise ValueError(f"Unknown channel name: {channel}")
        channel_idx = channel_map[channel]
    else:
        channel_idx = channel

    if channel_idx >= image.shape[2]:
        raise ValueError(
            f"Channel {channel_idx} not in image with {image.shape[2]} channels"
        )

    return image[:, :, channel_idx]
