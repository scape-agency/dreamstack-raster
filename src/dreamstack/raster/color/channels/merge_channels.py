"""Merge channels operation."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def merge_channels(
    channels: list[NDArray[np.uint8]] | tuple[NDArray[np.uint8], ...],
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
