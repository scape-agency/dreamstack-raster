# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Radial Blur
===============================

Radial blur filter implementation.

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


def radial_blur(
    image: Image,
    amount: float = 10,
    center: tuple[float, float] | None = None,
    mode: str = "spin",
) -> Image:
    """
    Apply radial blur (spin or zoom).

    Args:
        image: Input image
        amount: Blur amount
        center: Blur center (relative 0-1), defaults to image center
        mode: 'spin' for rotational, 'zoom' for linear

    Returns:
        Blurred image
    """
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.pixel import PixelData

    h, w = image.height, image.width

    if center is None:
        cx, cy = 0.5, 0.5
    else:
        cx, cy = center

    cx *= w
    cy *= h

    # Create coordinate grids
    y, x = np.ogrid[:h, :w]

    data = image.data.astype(np.float32)

    # Number of samples for motion blur effect
    samples = int(amount)
    result = np.zeros_like(data)

    if mode == "spin":
        # Rotational blur
        for i in range(samples):
            angle = (i - samples / 2) * (amount / samples) * np.pi / 180

            # Rotation around center
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)

            new_x = cos_a * (x - cx) - sin_a * (y - cy) + cx
            new_y = sin_a * (x - cx) + cos_a * (y - cy) + cy

            # Clamp coordinates
            new_x = np.clip(new_x, 0, w - 1).astype(np.float32)
            new_y = np.clip(new_y, 0, h - 1).astype(np.float32)

            # Sample
            if data.ndim == 3:
                for c in range(data.shape[2]):
                    result[:, :, c] += cv2.remap(
                        data[:, :, c], new_x, new_y, cv2.INTER_LINEAR
                    )
            else:
                result += cv2.remap(data, new_x, new_y, cv2.INTER_LINEAR)
    else:
        # Zoom blur
        for i in range(samples):
            scale = 1 + (i - samples / 2) * (amount / samples) / 100

            new_x = (x - cx) * scale + cx
            new_y = (y - cy) * scale + cy

            new_x = np.clip(new_x, 0, w - 1).astype(np.float32)
            new_y = np.clip(new_y, 0, h - 1).astype(np.float32)

            if data.ndim == 3:
                for c in range(data.shape[2]):
                    result[:, :, c] += cv2.remap(
                        data[:, :, c], new_x, new_y, cv2.INTER_LINEAR
                    )
            else:
                result += cv2.remap(data, new_x, new_y, cv2.INTER_LINEAR)

    result /= samples

    result_image = image.copy()
    result_image._pixel_data = PixelData(  # pylint: disable=protected-access
        data=result.astype(image.data.dtype),
        pixel_format=image.pixel_format,
        bit_depth=image.bit_depth,
    )

    return result_image
