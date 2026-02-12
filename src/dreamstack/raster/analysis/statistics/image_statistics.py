"""
Image Statistics Function
=========================

Compute comprehensive image statistics.

"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .channel_statistics import channel_statistics
from .image_stats import ImageStats


def image_statistics(
    image: NDArray[np.uint8],
    mask: NDArray[np.uint8] | None = None,
) -> ImageStats:
    """Compute comprehensive image statistics.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input image.
    mask : NDArray[np.uint8], optional
        Optional mask.

    Returns
    -------
    ImageStats
        Image statistics.
    """
    h, w = image.shape[:2]
    channels = image.shape[2] if image.ndim == 3 else 1

    channel_stats = []
    for c in range(channels):
        stats = channel_statistics(image, c, mask)
        channel_stats.append(stats)

    return ImageStats(
        width=w,
        height=h,
        channels=channels,
        dtype=str(image.dtype),
        channel_stats=channel_stats,
    )
