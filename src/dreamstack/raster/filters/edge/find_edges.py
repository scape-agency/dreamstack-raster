# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Find Edges
==============================

Photoshop-style Find Edges filter implementation.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.image import Image


def find_edges(image: Image) -> Image:
    """
    Apply Photoshop-style Find Edges filter.

    Args:
        image: Input image

    Returns:
        Inverted edge image
    """
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.pixel import PixelData

    data = image.data.astype(np.float32)
    max_val = 255 if image.bit_depth.name == "UINT8" else 65535

    if data.ndim == 3:
        # Process each channel
        result = np.zeros_like(data)
        for i in range(data.shape[2]):
            sobelx = cv2.Sobel(data[:, :, i], cv2.CV_32F, 1, 0, ksize=3)
            sobely = cv2.Sobel(data[:, :, i], cv2.CV_32F, 0, 1, ksize=3)
            magnitude = np.sqrt(sobelx**2 + sobely**2)
            result[:, :, i] = max_val - np.clip(magnitude, 0, max_val)
    else:
        sobelx = cv2.Sobel(data, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(data, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        result = max_val - np.clip(magnitude, 0, max_val)

    result_image = image.copy()
    result_image._pixel_data = PixelData(  # pylint: disable=protected-access
        data=result.astype(image.data.dtype),
        pixel_format=image.pixel_format,
        bit_depth=image.bit_depth,
    )

    return result_image
