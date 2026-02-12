"""Extract RGB arrays as colorized images."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from .split_channels import split_channels


def extract_rgb_arrays(
    image: NDArray[np.uint8],
    color_format: str = "BGR",
) -> tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8]]:
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
