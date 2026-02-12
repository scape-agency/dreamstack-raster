"""
Channel Statistics Function
===========================

Compute statistics for a single channel.

"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .channel_stats import ChannelStats


def channel_statistics(
    image: NDArray[np.uint8],
    channel: int = 0,
    mask: NDArray[np.uint8] | None = None,
) -> ChannelStats:
    """Compute statistics for a single channel.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input image.
    channel : int, optional
        Channel index. Default is 0.
    mask : NDArray[np.uint8], optional
        Optional mask.

    Returns
    -------
    ChannelStats
        Channel statistics.
    """
    if image.ndim == 3:
        data = image[:, :, channel]
    else:
        data = image

    if mask is not None:
        data = data[mask > 0]
    else:
        data = data.flatten()

    return ChannelStats(
        mean=float(np.mean(data)),
        std=float(np.std(data)),
        min=int(np.min(data)),
        max=int(np.max(data)),
        median=float(np.median(data)),
    )
