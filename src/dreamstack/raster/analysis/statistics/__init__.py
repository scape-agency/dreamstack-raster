# -*- coding: utf-8 -*-

"""
Image Statistics Module
=======================

Statistical analysis functions for images.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray


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
    channel_stats: List[ChannelStats]


def channel_statistics(
    image: NDArray[np.uint8],
    channel: int = 0,
    mask: Optional[NDArray[np.uint8]] = None,
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


def image_statistics(
    image: NDArray[np.uint8],
    mask: Optional[NDArray[np.uint8]] = None,
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


def color_count(
    image: NDArray[np.uint8],
    mask: Optional[NDArray[np.uint8]] = None,
) -> int:
    """Count unique colors in image.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input image.
    mask : NDArray[np.uint8], optional
        Optional mask.

    Returns
    -------
    int
        Number of unique colors.
    """
    if mask is not None:
        if image.ndim == 3:
            pixels = image[mask > 0]
        else:
            pixels = image[mask > 0]
    else:
        if image.ndim == 3:
            pixels = image.reshape(-1, image.shape[2])
        else:
            pixels = image.flatten()

    if image.ndim == 3:
        # Convert to single value per pixel
        unique = np.unique(pixels.astype(np.int64).dot([1, 256, 65536]))
        return len(unique)
    else:
        return len(np.unique(pixels))


def unique_colors(
    image: NDArray[np.uint8],
    max_colors: int = 1000,
    mask: Optional[NDArray[np.uint8]] = None,
) -> List[Tuple[int, ...]]:
    """Get list of unique colors.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input image.
    max_colors : int, optional
        Maximum colors to return. Default is 1000.
    mask : NDArray[np.uint8], optional
        Optional mask.

    Returns
    -------
    list
        List of unique colors as tuples.
    """
    if mask is not None:
        if image.ndim == 3:
            pixels = image[mask > 0]
        else:
            pixels = image[mask > 0]
    else:
        if image.ndim == 3:
            pixels = image.reshape(-1, image.shape[2])
        else:
            pixels = image.flatten()

    unique = np.unique(pixels, axis=0)

    if len(unique) > max_colors:
        unique = unique[:max_colors]

    if image.ndim == 3:
        return [tuple(int(v) for v in c) for c in unique]
    else:
        return [(int(c),) for c in unique]
