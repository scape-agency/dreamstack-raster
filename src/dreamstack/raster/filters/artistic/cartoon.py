# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Cartoon
===========================

Cartoon/cel-shading effect implementation.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

# =============================================================================
# Type Checking Imports
# =============================================================================

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.image import Image


def cartoon(
    image: Image,
    edge_threshold: float = 100,
    color_levels: int = 8,
) -> Image:
    """
    Apply cartoon/cel-shading effect.

    Args:
        image: Input image
        edge_threshold: Edge detection threshold
        color_levels: Number of color levels

    Returns:
        Cartoon-style image
    """
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.pixel import PixelData

    data = image.data

    if image.bit_depth.name != "UINT8":
        data = (data / data.max() * 255).astype(np.uint8)

    if data.ndim == 3 and data.shape[2] >= 3:
        # Bilateral filter for smoothing
        color = cv2.bilateralFilter(data[:, :, :3], 9, 75, 75)

        # Reduce colors
        div = 256 // color_levels
        color = (color // div) * div

        # Get edges
        gray = cv2.cvtColor(data[:, :, :3], cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur to reduce noise before edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Use Canny edge detection with edge_threshold
        canny_edges = cv2.Canny(blurred, edge_threshold / 2, edge_threshold)
        # Invert edges and dilate slightly for better lines
        edges = 255 - cv2.dilate(canny_edges, np.ones((2, 2), np.uint8))

        # Combine
        edges_3d = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        result = cv2.bitwise_and(color, edges_3d)

        if data.shape[2] == 4:
            result = np.dstack([result, data[:, :, 3]])
    else:
        if data.ndim == 3:
            data = data[:, :, 0]
        result = cv2.bilateralFilter(data, 9, 75, 75)
        div = 256 // color_levels
        result = (result // div) * div
        result = result[:, :, np.newaxis]

    if image.bit_depth.name != "UINT8":
        max_val = 65535 if image.bit_depth.name == "UINT16" else 1.0
        result = (result / 255.0 * max_val).astype(image.data.dtype)

    result_image = image.copy()
    result_image._pixel_data = PixelData(  # pylint: disable=protected-access
        data=result, pixel_format=image.pixel_format, bit_depth=image.bit_depth
    )

    return result_image
