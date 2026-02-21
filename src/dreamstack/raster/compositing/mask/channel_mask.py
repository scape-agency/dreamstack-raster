# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Channel Mask
============

Create masks from color channels.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import cv2
import numpy as np

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel
    from numpy.typing import NDArray


def channel_mask(
    image: NDArray[np.uint8],
    channel: Literal[
        "red", "green", "blue", "alpha", "luminosity", "saturation"
    ] = "red",
    *,
    invert: bool = False,
) -> NDArray[np.uint8]:
    """Create a mask from a specific color channel.

    Extracts a single channel and uses it as a mask.
    Useful for creating selections based on color content.

    Args:
        image: Input image.
        channel: Which channel to use.
        invert: If True, invert the resulting mask.

    Returns:
        Grayscale mask from channel.

    Example:
        >>> # Use red channel as mask
        >>> red_mask = channel_mask(image, "red")
        >>> # Mask based on saturation
        >>> sat_mask = channel_mask(image, "saturation")
    """
    h, w = image.shape[:2]

    if channel == "red":
        if image.ndim == 2:
            mask = image
        else:
            mask = image[:, :, 2]  # BGR order

    elif channel == "green":
        if image.ndim == 2:
            mask = image
        else:
            mask = image[:, :, 1]

    elif channel == "blue":
        if image.ndim == 2:
            mask = image
        else:
            mask = image[:, :, 0]

    elif channel == "alpha":
        if image.ndim > 2 and image.shape[2] == 4:
            mask = image[:, :, 3]
        else:
            mask = np.full((h, w), 255, dtype=np.uint8)

    elif channel == "luminosity":
        if image.ndim == 2:
            mask = image
        else:
            rgb = image[:, :, :3].astype(np.float32)
            mask = (
                0.299 * rgb[:, :, 2]
                + 0.587 * rgb[:, :, 1]
                + 0.114 * rgb[:, :, 0]
            )
            mask = mask.astype(np.uint8)

    elif channel == "saturation":
        if image.ndim == 2:
            mask = np.zeros((h, w), dtype=np.uint8)
        else:
            hsv = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2HSV)
            mask = hsv[:, :, 1]

    else:
        raise ValueError(f"Unknown channel: {channel}")

    if invert:
        mask = 255 - mask

    return mask.copy()
