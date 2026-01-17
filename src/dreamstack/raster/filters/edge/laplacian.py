# -*- coding: utf-8 -*-

"""
Dreamstack Raster - Laplacian Edge Detection
============================================

Laplacian edge detection implementation.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from dreamstack.raster.core.image import Image


def laplacian(image: Image, ksize: int = 3, scale: float = 1.0) -> Image:
    """
    Apply Laplacian edge detection.

    Args:
        image: Input image
        ksize: Kernel size
        scale: Scale factor

    Returns:
        Edge image
    """
    from dreamstack.raster.core.pixel import PixelData

    gray_img = image.to_grayscale() if image.channels > 1 else image
    data = gray_img.data.astype(np.float32)

    if data.ndim == 3:
        data = data[:, :, 0]

    laplacian_result = cv2.Laplacian(data, cv2.CV_32F, ksize=ksize) * scale

    # Convert to absolute values
    laplacian_result = np.abs(laplacian_result)

    max_val = 255 if image.bit_depth.name == "UINT8" else 65535
    laplacian_result = np.clip(laplacian_result, 0, max_val)

    result_image = image.copy()
    result_image._pixel_data = PixelData(
        data=laplacian_result[:, :, np.newaxis].astype(image.data.dtype),
        pixel_format=gray_img.pixel_format,
        bit_depth=image.bit_depth,
    )

    return result_image
