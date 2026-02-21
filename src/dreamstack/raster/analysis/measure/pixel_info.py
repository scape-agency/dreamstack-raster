# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Pixel Info Function
===================

Get detailed information about a pixel.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from .pixel_info_dataclass import PixelInfo
from .sample_color import sample_color


def pixel_info(
    image: NDArray[np.uint8],
    x: int,
    y: int,
) -> PixelInfo:
    """Get detailed information about a pixel.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input image (BGR).
    x : int
        X coordinate.
    y : int
        Y coordinate.

    Returns
    -------
    PixelInfo
        Detailed pixel information.
    """
    bgr = sample_color(image, x, y)
    rgb = (bgr[2], bgr[1], bgr[0])

    # Convert to HSV
    pixel = np.array([[bgr]], dtype=np.uint8)
    hsv_pixel = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)
    hsv_px = hsv_pixel[0, 0]
    hsv: tuple[int, int, int] = (
        int(hsv_px[0]),
        int(hsv_px[1]),
        int(hsv_px[2]),
    )

    # Convert to LAB
    lab_pixel = cv2.cvtColor(pixel, cv2.COLOR_BGR2LAB)
    lab_px = lab_pixel[0, 0]
    lab: tuple[int, int, int] = (
        int(lab_px[0]),
        int(lab_px[1]),
        int(lab_px[2]),
    )

    return PixelInfo(x=x, y=y, rgb=rgb, hsv=hsv, lab=lab)
