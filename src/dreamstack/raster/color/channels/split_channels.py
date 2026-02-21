# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""Split channels operation."""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def split_channels(
    image: NDArray[np.uint8],
    color_format: str = "BGR",
) -> tuple[NDArray[np.uint8], ...]:
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
            return (channels[2], channels[1], channels[0])  # type: ignore[return-value]
        elif len(channels) == 4:
            # type: ignore[return-value]
            return (channels[2], channels[1], channels[0], channels[3])

    return tuple(channels)  # type: ignore[return-value]
