"""
Dreamstack Raster - Sobel Edge Detection
========================================

Sobel edge detection implementation.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from dreamstack.raster.core.image import Image


def sobel(image: Image, ksize: int = 3, scale: float = 1.0) -> Image:
    """
    Apply Sobel edge detection.

    Args:
        image: Input image
        ksize: Kernel size (1, 3, 5, or 7)
        scale: Scale factor for result

    Returns:
        Edge image
    """
    from dreamstack.raster.core.pixel import PixelData

    # Convert to grayscale
    gray_img = image.to_grayscale() if image.channels > 1 else image
    data = gray_img.data.astype(np.float32)

    if data.ndim == 3:
        data = data[:, :, 0]

    # Sobel in X and Y
    sobelx = cv2.Sobel(data, cv2.CV_32F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(data, cv2.CV_32F, 0, 1, ksize=ksize)

    # Magnitude
    magnitude = np.sqrt(sobelx**2 + sobely**2) * scale

    # Normalize
    max_val = 255 if image.bit_depth.name == "UINT8" else 65535
    magnitude = np.clip(magnitude, 0, max_val)

    result_image = image.copy()
    result_image._pixel_data = PixelData(
        data=magnitude[:, :, np.newaxis].astype(image.data.dtype),
        pixel_format=gray_img.pixel_format,
        bit_depth=image.bit_depth,
    )

    return result_image
