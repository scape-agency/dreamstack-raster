# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Trace Contour
=================================

Trace contours implementation.

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


def trace_contour(
    image: Image,
    edge_mode: str = "upper",  # pylint: disable=unused-argument  # TODO: implement
) -> Image:
    """
    Trace contours in image.

    Args:
        image: Input image
        edge_mode: 'upper' or 'lower' edge tracing

    Returns:
        Contour-traced image
    """
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.pixel import PixelData

    gray = image.to_grayscale().data
    if gray.ndim == 3:
        gray = gray[:, :, 0]

    if gray.dtype != np.uint8:
        gray = (gray / gray.max() * 255).astype(np.uint8)

    max_val = 255 if image.bit_depth.name == "UINT8" else 65535

    # Threshold and find contours
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    result = np.ones_like(gray, dtype=np.uint8) * 255
    cv2.drawContours(result, contours, -1, 0, 1)

    # Convert back
    result = result.astype(np.float32) / 255 * max_val

    result_image = image.copy()
    result_image._pixel_data = PixelData(  # pylint: disable=protected-access
        data=result[:, :, np.newaxis].astype(image.data.dtype),
        pixel_format=image.to_grayscale().pixel_format,
        bit_depth=image.bit_depth,
    )

    return result_image
