# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Denoise NL-Means
====================================

Non-local means denoising implementation.

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


def denoise_nlmeans(
    image: Image,
    h: float = 10,
    template_window_size: int = 7,
    search_window_size: int = 21,
) -> Image:
    """
    Denoise using non-local means algorithm.

    Args:
        image: Input image
        h: Filter strength (higher removes more noise but less detail)
        template_window_size: Size of template patch
        search_window_size: Size of search window

    Returns:
        Denoised image
    """
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.pixel import PixelData

    data = image.data

    # Convert to 8-bit for cv2.fastNlMeans
    if image.bit_depth.name != "UINT8":
        data = (data / data.max() * 255).astype(np.uint8)

    if data.ndim == 3 and data.shape[2] >= 3:
        result = cv2.fastNlMeansDenoisingColored(
            data[:, :, :3],
            None,
            h,
            h,
            template_window_size,
            search_window_size,
        )
        if data.shape[2] == 4:
            result = np.dstack([result, data[:, :, 3]])
    else:
        if data.ndim == 3:
            data = data[:, :, 0]
        result = cv2.fastNlMeansDenoising(
            data, None, h, template_window_size, search_window_size
        )
        result = result[:, :, np.newaxis]

    # Convert back
    if image.bit_depth.name != "UINT8":
        max_val = 65535 if image.bit_depth.name == "UINT16" else 1.0
        result = (result / 255.0 * max_val).astype(image.data.dtype)

    result_image = image.copy()
    result_image._pixel_data = PixelData(  # pylint: disable=protected-access
        data=result, pixel_format=image.pixel_format, bit_depth=image.bit_depth
    )

    return result_image
