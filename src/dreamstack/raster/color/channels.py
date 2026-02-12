# -*- coding: utf-8 -*-

"""
Color Channel Operations
========================

Operations for splitting, extracting, and merging color channels
from images. Essential for machine learning preprocessing and
image analysis tasks.

"""

from __future__ import annotations

from typing import List, Literal, Optional, Tuple, Union

import cv2
import numpy as np
from numpy.typing import NDArray


ChannelName = Literal["red", "green", "blue", "alpha", "r", "g", "b", "a"]


def split_channels(
    image: NDArray[np.uint8],
    color_format: str = "BGR",
) -> Tuple[NDArray[np.uint8], ...]:
    """Split image into individual color channels.

    Separates a color image into its component channels,
    useful for channel-specific processing in ML pipelines.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input color image (H, W, C).
    color_format : str, optional
        Color format of input: "BGR" (OpenCV default), "RGB".
        Default is "BGR".

    Returns
    -------
    tuple[NDArray[np.uint8], ...]
        Tuple of single-channel images. For RGB/BGR: (R, G, B).
        For RGBA/BGRA: (R, G, B, A).

    Examples
    --------
    >>> import cv2
    >>> img = cv2.imread('image.jpg')
    >>> r, g, b = split_channels(img)
    >>> print(f"Red channel shape: {r.shape}")
    
    >>> # Display separated channels
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    >>> ax[0].imshow(r, cmap='Reds')
    >>> ax[1].imshow(g, cmap='Greens')
    >>> ax[2].imshow(b, cmap='Blues')
    """
    if image.ndim != 3:
        raise ValueError("Image must have 3 dimensions (H, W, C)")

    channels = cv2.split(image)

    if color_format.upper() in ("BGR", "BGRA"):
        # Convert from BGR to RGB order
        if len(channels) == 3:
            return (channels[2], channels[1], channels[0])
        elif len(channels) == 4:
            return (channels[2], channels[1], channels[0], channels[3])

    return tuple(channels)


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
            "blue": 0, "b": 0,
            "green": 1, "g": 1,
            "red": 2, "r": 2,
            "alpha": 3, "a": 3,
        }
        channel_map_rgb = {
            "red": 0, "r": 0,
            "green": 1, "g": 1,
            "blue": 2, "b": 2,
            "alpha": 3, "a": 3,
        }
        channel_map = channel_map_bgr if color_format.upper() in ("BGR", "BGRA") else channel_map_rgb
        if channel not in channel_map:
            raise ValueError(f"Unknown channel name: {channel}")
        channel_idx = channel_map[channel]
    else:
        channel_idx = channel

    if channel_idx >= image.shape[2]:
        raise ValueError(f"Channel {channel_idx} not in image with {image.shape[2]} channels")

    return image[:, :, channel_idx]


def merge_channels(
    channels: Union[List[NDArray[np.uint8]], Tuple[NDArray[np.uint8], ...]],
    color_format: str = "BGR",
) -> NDArray[np.uint8]:
    """Merge individual channels into a color image.

    Combines separate channel arrays back into a multi-channel image.

    Parameters
    ----------
    channels : list or tuple of NDArray[np.uint8]
        List of single-channel images (R, G, B) or (R, G, B, A).
    color_format : str, optional
        Output color format: "BGR" or "RGB". Default is "BGR".

    Returns
    -------
    NDArray[np.uint8]
        Merged color image (H, W, C).

    Examples
    --------
    >>> r, g, b = split_channels(img)
    >>> # Enhance red channel
    >>> r_enhanced = cv2.multiply(r, 1.2)
    >>> result = merge_channels([r_enhanced, g, b])
    """
    if len(channels) < 3:
        raise ValueError("Need at least 3 channels (R, G, B)")

    if color_format.upper() in ("BGR", "BGRA"):
        # Convert from RGB order to BGR
        if len(channels) == 3:
            return cv2.merge([channels[2], channels[1], channels[0]])
        elif len(channels) >= 4:
            return cv2.merge([channels[2], channels[1], channels[0], channels[3]])

    return cv2.merge(list(channels))


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
        channel_map_bgr = {"blue": 0, "b": 0, "green": 1, "g": 1, "red": 2, "r": 2}
        channel_map_rgb = {"red": 0, "r": 0, "green": 1, "g": 1, "blue": 2, "b": 2}
        channel_map = channel_map_bgr if color_format.upper() in ("BGR", "BGRA") else channel_map_rgb
        channel_idx = channel_map.get(channel, 0)
    else:
        channel_idx = channel

    result[:, :, channel_idx] = image[:, :, channel_idx]
    return result


def extract_rgb_arrays(
    image: NDArray[np.uint8],
    color_format: str = "BGR",
) -> Tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8]]:
    """Extract R, G, B as separate colorized images.

    Creates three images where each shows only one color channel
    in its full color representation. Useful for visualization.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input color image (H, W, C).
    color_format : str, optional
        Color format: "BGR" or "RGB". Default is "BGR".

    Returns
    -------
    tuple[NDArray, NDArray, NDArray]
        (red_image, green_image, blue_image) - Each is (H, W, 3).

    Examples
    --------
    >>> r_img, g_img, b_img = extract_rgb_arrays(img)
    >>> # Display for ML visualization
    >>> plt.subplot(1, 3, 1); plt.imshow(cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB))
    >>> plt.subplot(1, 3, 2); plt.imshow(cv2.cvtColor(g_img, cv2.COLOR_BGR2RGB))
    >>> plt.subplot(1, 3, 3); plt.imshow(cv2.cvtColor(b_img, cv2.COLOR_BGR2RGB))
    """
    r, g, b = split_channels(image, color_format)

    # Create colorized versions
    zeros = np.zeros_like(r)

    if color_format.upper() in ("BGR", "BGRA"):
        red_img = cv2.merge([zeros, zeros, r])
        green_img = cv2.merge([zeros, g, zeros])
        blue_img = cv2.merge([b, zeros, zeros])
    else:
        red_img = cv2.merge([r, zeros, zeros])
        green_img = cv2.merge([zeros, g, zeros])
        blue_img = cv2.merge([zeros, zeros, b])

    return red_img, green_img, blue_img


def swap_channels(
    image: NDArray[np.uint8],
    from_channel: Union[int, ChannelName],
    to_channel: Union[int, ChannelName],
    color_format: str = "BGR",
) -> NDArray[np.uint8]:
    """Swap two color channels.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input color image.
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
    channel_map = channel_map_bgr if color_format.upper() in ("BGR", "BGRA") else channel_map_rgb

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


def channel_to_grayscale_rgb(
    channel: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    """Convert single channel to 3-channel grayscale.

    Creates RGB image where all channels have the same values.
    Useful for displaying single channels in color pipelines.

    Parameters
    ----------
    channel : NDArray[np.uint8]
        Single channel image (H, W).

    Returns
    -------
    NDArray[np.uint8]
        3-channel grayscale image (H, W, 3).
    """
    return cv2.merge([channel, channel, channel])
