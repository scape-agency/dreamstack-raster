# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Glass Distortion
====================================

Glass distortion effect filter implementation.

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


def glass(
    image: Image,
    distortion: float = 5,
    smoothness: int = 3,
    texture: np.ndarray | None = None,
) -> Image:
    """
    Apply glass distortion effect.

    Args:
        image: Input image
        distortion: Distortion amount
        smoothness: Texture smoothness
        texture: Optional displacement texture

    Returns:
        Distorted image
    """
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.pixel import PixelData

    data = image.data
    h, w = data.shape[:2]

    if texture is None:
        # Generate random noise texture
        texture = np.random.random((h, w)).astype(np.float32)

    # Smooth the texture
    if smoothness > 0:
        ksize = smoothness * 2 + 1
        texture = cv2.GaussianBlur(texture, (ksize, ksize), 0)

    # Create displacement maps
    y, x = np.mgrid[:h, :w].astype(np.float32)  # pylint: disable=no-member

    # Compute gradients
    grad_x = cv2.Sobel(texture, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(texture, cv2.CV_32F, 0, 1, ksize=3)

    map_x = x + grad_x * distortion
    map_y = y + grad_y * distortion

    result = cv2.remap(
        data, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
    )

    result_image = image.copy()
    result_image._pixel_data = PixelData(  # pylint: disable=protected-access
        data=result, pixel_format=image.pixel_format, bit_depth=image.bit_depth
    )

    return result_image
