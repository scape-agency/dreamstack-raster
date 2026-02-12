"""
Dreamstack Raster - Emboss
==========================

Emboss effect implementation.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from dreamstack.raster.core.image import Image


def emboss(
    image: Image, angle: float = 135, height: int = 1, amount: float = 100
) -> Image:
    """
    Apply emboss effect.

    Args:
        image: Input image
        angle: Light angle in degrees
        height: Emboss depth (1-10)
        amount: Effect amount (1-500%)

    Returns:
        Embossed image
    """
    from dreamstack.raster.core.pixel import PixelData

    data = image.data.astype(np.float32)
    max_val = 255 if image.bit_depth.name == "UINT8" else 65535
    mid_val = max_val / 2

    # Create emboss kernel based on angle
    angle_rad = np.radians(angle)
    dx = int(np.round(np.cos(angle_rad)))
    dy = int(np.round(np.sin(angle_rad)))

    kernel = np.zeros((3, 3), dtype=np.float32)
    kernel[1 - dy, 1 - dx] = -height
    kernel[1, 1] = 0
    kernel[1 + dy, 1 + dx] = height

    if data.ndim == 3:
        result = np.zeros_like(data)
        for i in range(data.shape[2]):
            embossed = cv2.filter2D(data[:, :, i], -1, kernel)
            # Add mid gray and apply amount
            result[:, :, i] = mid_val + embossed * (amount / 100)
    else:
        embossed = cv2.filter2D(data, -1, kernel)
        result = mid_val + embossed * (amount / 100)

    result = np.clip(result, 0, max_val)

    result_image = image.copy()
    result_image._pixel_data = PixelData(
        data=result.astype(image.data.dtype),
        pixel_format=image.pixel_format,
        bit_depth=image.bit_depth,
    )

    return result_image
