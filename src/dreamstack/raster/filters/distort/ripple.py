# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Ripple Distortion
=====================================

Ripple distortion filter implementation.

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


def ripple(
    image: Image,
    amplitude: float = 5,
    wavelength: float = 20,
    center: tuple[float, float] | None = None,
) -> Image:
    """
    Apply ripple distortion (concentric waves from center).

    Args:
        image: Input image
        amplitude: Wave amplitude
        wavelength: Distance between ripples
        center: Ripple center (relative 0-1)

    Returns:
        Distorted image
    """
    from dreamstack.raster.core.pixel import PixelData

    data = image.data
    h, w = data.shape[:2]

    if center is None:
        cx, cy = w / 2, h / 2
    else:
        cx, cy = center[0] * w, center[1] * h

    y, x = np.mgrid[:h, :w].astype(np.float32)  # pylint: disable=no-member

    # Distance from center
    dx = x - cx
    dy = y - cy
    distance = np.sqrt(dx**2 + dy**2)

    # Ripple offset
    offset = amplitude * np.sin(2 * np.pi * distance / wavelength)

    # Normalize direction
    with np.errstate(divide="ignore", invalid="ignore"):
        norm_dx = np.where(distance > 0, dx / distance, 0)
        norm_dy = np.where(distance > 0, dy / distance, 0)

    map_x = (x + norm_dx * offset).astype(np.float32)
    map_y = (y + norm_dy * offset).astype(np.float32)

    result = cv2.remap(
        data, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
    )

    result_image = image.copy()
    result_image._pixel_data = PixelData(  # pylint: disable=protected-access
        data=result, pixel_format=image.pixel_format, bit_depth=image.bit_depth
    )

    return result_image
