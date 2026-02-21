# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Canny Edge Detection
========================================

Canny edge detection implementation.

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


def canny(
    image: Image,
    threshold1: float = 100,
    threshold2: float = 200,
    aperture_size: int = 3,
    l2_gradient: bool = True,
) -> Image:
    """
    Apply Canny edge detection.

    Args:
        image: Input image
        threshold1: Lower threshold for hysteresis
        threshold2: Upper threshold for hysteresis
        aperture_size: Aperture size for Sobel operator
        l2_gradient: Use L2 norm for gradient magnitude

    Returns:
        Binary edge image
    """
    from dreamstack.raster.core.pixel import PixelData

    # Convert to grayscale 8-bit
    gray_img = image.to_grayscale() if image.channels > 1 else image
    data = gray_img.data

    if data.dtype != np.uint8:
        data = (data / data.max() * 255).astype(np.uint8)

    if data.ndim == 3:
        data = data[:, :, 0]

    edges = cv2.Canny(
        data,
        threshold1,
        threshold2,
        apertureSize=aperture_size,
        L2gradient=l2_gradient,
    )

    result_image = image.copy()
    result_image._pixel_data = PixelData(  # pylint: disable=protected-access
        data=edges[:, :, np.newaxis],
        pixel_format=gray_img.pixel_format,
        bit_depth=gray_img.bit_depth,
    )

    return result_image


def canny_auto(
    image: Image,
    sigma: float = 0.33,
    aperture_size: int = 3,
) -> Image:
    """
    Apply Canny edge detection with automatic threshold selection.

    Computes optimal thresholds based on median pixel intensity.

    Args:
        image: Input image
        sigma: Threshold adjustment factor (default 0.33)
        aperture_size: Aperture size for Sobel operator

    Returns:
        Binary edge image
    """
    from dreamstack.raster.core.pixel import PixelData

    # Convert to grayscale 8-bit
    gray_img = image.to_grayscale() if image.channels > 1 else image
    data = gray_img.data

    if data.dtype != np.uint8:
        data = (data / data.max() * 255).astype(np.uint8)

    if data.ndim == 3:
        data = data[:, :, 0]

    # Compute optimal thresholds from median
    median = np.median(data)
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))

    edges = cv2.Canny(
        data,
        lower,
        upper,
        apertureSize=aperture_size,
    )

    result_image = image.copy()
    result_image._pixel_data = PixelData(  # pylint: disable=protected-access
        data=edges[:, :, np.newaxis],
        pixel_format=gray_img.pixel_format,
        bit_depth=gray_img.bit_depth,
    )

    return result_image
