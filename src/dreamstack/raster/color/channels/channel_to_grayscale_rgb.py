"""Convert channel to grayscale RGB."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


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
