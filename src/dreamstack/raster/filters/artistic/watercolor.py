# -*- coding: utf-8 -*-

"""
Dreamstack Raster - Watercolor
==============================

Watercolor effect implementation.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from dreamstack.raster.core.image import Image


def watercolor(
    image: Image, smoothness: int = 5, edge_threshold: float = 0.5
) -> Image:
    """
    Apply watercolor effect.

    Args:
        image: Input image
        smoothness: Smoothing amount
        edge_threshold: Edge darkening threshold

    Returns:
        Stylized image
    """
    from dreamstack.raster.core.pixel import PixelData

    data = image.data

    if image.bit_depth.name != "UINT8":
        data = (data / data.max() * 255).astype(np.uint8)

    # Bilateral filter for smoothing while keeping edges
    if data.ndim == 3 and data.shape[2] >= 3:
        smoothed = cv2.bilateralFilter(data[:, :, :3], 9, 75, 75)

        # Apply multiple times for stronger effect
        for _ in range(smoothness - 1):
            smoothed = cv2.bilateralFilter(smoothed, 9, 75, 75)

        # Add edge darkening
        gray = cv2.cvtColor(smoothed, cv2.COLOR_BGR2GRAY)
        edges = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 5
        )

        # Combine
        edges_3d = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        result = cv2.bitwise_and(smoothed, edges_3d)

        if data.shape[2] == 4:
            result = np.dstack([result, data[:, :, 3]])
    else:
        if data.ndim == 3:
            data = data[:, :, 0]
        result = cv2.bilateralFilter(data, 9, 75, 75)
        for _ in range(smoothness - 1):
            result = cv2.bilateralFilter(result, 9, 75, 75)
        result = result[:, :, np.newaxis]

    if image.bit_depth.name != "UINT8":
        max_val = 65535 if image.bit_depth.name == "UINT16" else 1.0
        result = (result / 255.0 * max_val).astype(image.data.dtype)

    result_image = image.copy()
    result_image._pixel_data = PixelData(
        data=result, pixel_format=image.pixel_format, bit_depth=image.bit_depth
    )

    return result_image
