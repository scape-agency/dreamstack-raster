"""
Dreamstack Raster - Wave Distortion
===================================

Wave distortion filter implementation.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from dreamstack.raster.core.image import Image


def wave(
    image: Image,
    amplitude: float = 10,
    wavelength: float = 50,
    direction: str = "horizontal",
    wave_type: str = "sine",
) -> Image:
    """
    Apply wave distortion.

    Args:
        image: Input image
        amplitude: Wave amplitude in pixels
        wavelength: Wave length in pixels
        direction: 'horizontal' or 'vertical'
        wave_type: 'sine' or 'triangle'

    Returns:
        Distorted image
    """
    from dreamstack.raster.core.pixel import PixelData

    data = image.data
    h, w = data.shape[:2]

    # Create coordinate maps
    y, x = np.mgrid[:h, :w].astype(np.float32)

    if wave_type == "sine":
        if direction == "horizontal":
            offset = amplitude * np.sin(2 * np.pi * y / wavelength)
            map_x = x + offset
            map_y = y
        else:
            offset = amplitude * np.sin(2 * np.pi * x / wavelength)
            map_x = x
            map_y = y + offset
    else:  # triangle
        if direction == "horizontal":
            offset = amplitude * (2 * np.abs((y / wavelength) % 1 - 0.5) - 0.5) * 2
            map_x = x + offset
            map_y = y
        else:
            offset = amplitude * (2 * np.abs((x / wavelength) % 1 - 0.5) - 0.5) * 2
            map_x = x
            map_y = y + offset

    # Remap
    if data.ndim == 3:
        result = cv2.remap(
            data, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        )
    else:
        result = cv2.remap(
            data, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        )

    result_image = image.copy()
    result_image._pixel_data = PixelData(
        data=result, pixel_format=image.pixel_format, bit_depth=image.bit_depth
    )

    return result_image
