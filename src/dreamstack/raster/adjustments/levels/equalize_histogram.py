# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""Histogram equalization function."""


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


def equalize_histogram(image: Image) -> Image:
    """
    Equalize histogram for enhanced contrast.

    Args:
        image: Input image

    Returns:
        Equalized image
    """
    from dreamstack.raster.core.pixel import PixelData

    data = image.data

    # Convert to 8-bit for histogram equalization
    if image.bit_depth.name != "UINT8":
        data = (data / data.max() * 255).astype(np.uint8)

    if data.ndim == 3:
        # Convert to LAB and equalize L channel
        if data.shape[2] >= 3:
            lab = cv2.cvtColor(data[:, :, :3], cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            if data.shape[2] == 4:
                result = np.dstack([result, data[:, :, 3]])
        else:
            result = cv2.equalizeHist(data[:, :, 0])[:, :, np.newaxis]
    else:
        result = cv2.equalizeHist(data)

    # Convert back
    if image.bit_depth.name != "UINT8":
        max_val = 65535 if image.bit_depth.name == "UINT16" else 1.0
        result = (result / 255.0 * max_val).astype(image.data.dtype)

    result_image = image.copy()
    result_image._pixel_data = PixelData(  # pylint: disable=protected-access
        data=result, pixel_format=image.pixel_format, bit_depth=image.bit_depth
    )

    return result_image
