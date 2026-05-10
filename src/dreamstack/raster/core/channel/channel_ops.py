# -*- coding: utf-8 -*-

# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Channel Operations
=======================================

Pure-numpy helpers for splitting, merging, isolating, and swapping the
color channels of an image array.

These operations follow the project conventions:

- Channel order is **RGBA** (no BGR). Inputs assumed to be in RGB[A] order.
- Operations are dtype-agnostic — they preserve the dtype of the input
  (uint8, uint16, float32, …).
- 2-D grayscale arrays are accepted and treated as a single-channel image.

For the image-model concept of a "channel" (named, typed, addressable on
``Image``), see :class:`dreamstack.raster.core.channel.Channel`.
"""

# =============================================================================
# Imports
# =============================================================================

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

ChannelName = Literal["red", "green", "blue", "alpha", "r", "g", "b", "a"]

_NAME_TO_INDEX: dict[str, int] = {
    "red": 0,
    "r": 0,
    "green": 1,
    "g": 1,
    "blue": 2,
    "b": 2,
    "alpha": 3,
    "a": 3,
}


def _resolve(channel: int | ChannelName, n_channels: int) -> int:
    if isinstance(channel, str):
        try:
            idx = _NAME_TO_INDEX[channel.lower()]
        except KeyError as exc:
            raise ValueError(f"Unknown channel name: {channel!r}") from exc
    else:
        idx = int(channel)
    if not 0 <= idx < n_channels:
        raise ValueError(
            f"Channel index {idx} out of range for image with "
            f"{n_channels} channels"
        )
    return idx


def _ensure_3d(image: NDArray) -> NDArray:
    if image.ndim == 2:
        return image[:, :, np.newaxis]
    if image.ndim != 3:
        raise ValueError(
            f"Image must be 2D or 3D (H,W[,C]); got ndim={image.ndim}"
        )
    return image


# -----------------------------------------------------------------------------
# Split / merge
# -----------------------------------------------------------------------------


def split_channels(image: NDArray) -> tuple[NDArray, ...]:
    """Split an RGB(A) image into a tuple of single-channel 2-D arrays.

    The returned arrays are *views* into the input. Channels are returned
    in storage order (R, G, B[, A]).
    """
    arr = _ensure_3d(image)
    return tuple(arr[:, :, i] for i in range(arr.shape[2]))


def merge_channels(channels: list[NDArray] | tuple[NDArray, ...]) -> NDArray:
    """Stack single-channel arrays back into an RGB(A) image.

    Channels are stacked in the order given (R, G, B[, A]).
    """
    if len(channels) < 1:
        raise ValueError("merge_channels needs at least one channel")
    return np.stack(list(channels), axis=-1)


# -----------------------------------------------------------------------------
# Extract / isolate / swap
# -----------------------------------------------------------------------------


def extract_channel(image: NDArray, channel: int | ChannelName) -> NDArray:
    """Return a single channel as a 2-D array (view into ``image``)."""
    arr = _ensure_3d(image)
    return arr[:, :, _resolve(channel, arr.shape[2])]


def isolate_channel(image: NDArray, channel: int | ChannelName) -> NDArray:
    """Return an image with all channels but ``channel`` zeroed.

    Alpha (if present) is preserved untouched.
    """
    arr = _ensure_3d(image)
    idx = _resolve(channel, arr.shape[2])
    out = np.zeros_like(arr)
    out[:, :, idx] = arr[:, :, idx]
    if arr.shape[2] == 4 and idx != 3:
        out[:, :, 3] = arr[:, :, 3]
    return out


def swap_channels(
    image: NDArray,
    from_channel: int | ChannelName,
    to_channel: int | ChannelName,
) -> NDArray:
    """Return a copy of ``image`` with two channels swapped."""
    arr = _ensure_3d(image)
    i = _resolve(from_channel, arr.shape[2])
    j = _resolve(to_channel, arr.shape[2])
    out = arr.copy()
    out[:, :, i] = arr[:, :, j]
    out[:, :, j] = arr[:, :, i]
    return out


# -----------------------------------------------------------------------------
# Visualization helpers
# -----------------------------------------------------------------------------


def channel_to_grayscale_rgb(channel: NDArray) -> NDArray:
    """Promote a single channel (H, W) to a 3-channel grayscale RGB image."""
    if channel.ndim != 2:
        raise ValueError(
            f"channel must be 2D (H,W); got shape {channel.shape}"
        )
    return np.stack([channel, channel, channel], axis=-1)


def extract_rgb_arrays(
    image: NDArray,
) -> tuple[NDArray, NDArray, NDArray]:
    """Return ``(red_image, green_image, blue_image)`` colorized views.

    Each returned image keeps only one of R/G/B and zeros the others, in
    the original input dtype. Alpha is dropped.
    """
    arr = _ensure_3d(image)
    if arr.shape[2] < 3:
        raise ValueError(
            "extract_rgb_arrays requires an RGB(A) image with >=3 channels"
        )
    rgb = arr[:, :, :3]
    zeros = np.zeros_like(rgb[:, :, 0])
    red = np.stack([rgb[:, :, 0], zeros, zeros], axis=-1)
    green = np.stack([zeros, rgb[:, :, 1], zeros], axis=-1)
    blue = np.stack([zeros, zeros, rgb[:, :, 2]], axis=-1)
    return red, green, blue
