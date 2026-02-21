# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Contour
===========================

Contour effect (like topographic maps) implementation.

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
    from dreamstack.raster.core.image import Image


def contour(image: Image, levels: int = 8, edge_width: int = 1) -> Image:
    """
    Apply contour effect (like topographic maps).

    Args:
        image: Input image
        levels: Number of contour levels
        edge_width: Width of contour lines

    Returns:
        Contoured image
    """
    from dreamstack.raster.core.pixel import PixelData

    gray = image.to_grayscale().data.astype(np.float32)
    if gray.ndim == 3:
        gray = gray[:, :, 0]

    max_val = 255 if image.bit_depth.name == "UINT8" else 65535

    # Quantize to levels
    step = max_val / levels
    quantized = np.floor(gray / step) * step

    # Find edges between levels
    edges = np.zeros_like(gray)

    # Horizontal edges
    h_diff = np.abs(np.diff(quantized, axis=1))
    edges[:, :-1] += h_diff
    edges[:, 1:] += h_diff

    # Vertical edges
    v_diff = np.abs(np.diff(quantized, axis=0))
    edges[:-1, :] += v_diff
    edges[1:, :] += v_diff

    # Threshold edges
    result = np.where(edges > 0, 0, max_val)

    # Thicken edges
    if edge_width > 1:
        kernel = np.ones((edge_width, edge_width), np.uint8)
        result = cv2.erode(result.astype(np.uint8), kernel).astype(gray.dtype)

    result_image = image.copy()
    result_image._pixel_data = PixelData(  # pylint: disable=protected-access
        data=result[:, :, np.newaxis].astype(image.data.dtype),
        pixel_format=image.to_grayscale().pixel_format,
        bit_depth=image.bit_depth,
    )

    return result_image
