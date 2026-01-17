# -*- coding: utf-8 -*-

"""Dehaze function."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from dreamstack.raster.core.image import Image


def dehaze(image: Image, amount: float = 50) -> Image:
    """
    Remove haze/fog from image.

    Args:
        image: Input image
        amount: Dehaze strength (-100 to 100)

    Returns:
        Dehazed image
    """
    from dreamstack.raster.core.pixel import PixelData

    if image.channels < 3:
        return image.copy()

    data = image.data.astype(np.float32)
    max_val = 255 if image.bit_depth.name == "UINT8" else 65535

    normalized = data[:, :, :3] / max_val

    # Dark channel prior for haze detection
    # Minimum across color channels, then minimum filter
    min_channel = np.min(normalized, axis=2)

    ksize = 15
    dark_channel = cv2.erode(min_channel, np.ones((ksize, ksize)))

    # Estimate atmospheric light (brightest dark channel pixels)
    flat = dark_channel.flatten()
    num_bright = max(1, int(len(flat) * 0.001))
    indices = np.argpartition(flat, -num_bright)[-num_bright:]

    bright_pixels = normalized.reshape(-1, 3)[indices]
    atmospheric = np.max(bright_pixels, axis=0)

    # Transmission estimate
    omega = 0.95  # Keep some haze for realism
    transmission = 1 - omega * np.min(normalized / atmospheric, axis=2)

    # Refine transmission
    transmission = cv2.GaussianBlur(transmission, (15, 15), 0)
    transmission = np.clip(transmission, 0.1, 1)

    # Apply dehazing
    strength = amount / 100

    result = normalized.copy()

    if amount > 0:
        # Remove haze
        for i in range(3):
            result[:, :, i] = (
                normalized[:, :, i]
                - atmospheric[i] * (1 - transmission) * strength
            ) / np.maximum(transmission, 0.1)
    else:
        # Add haze
        for i in range(3):
            result[:, :, i] = normalized[:, :, i] * (
                1 + amount / 100
            ) + atmospheric[i] * (1 - transmission) * (-amount / 100)

    result = np.clip(result, 0, 1)

    final = data.copy()
    final[:, :, :3] = result * max_val

    result_image = image.copy()
    result_image._pixel_data = PixelData(
        data=final.astype(image.data.dtype),
        pixel_format=image.pixel_format,
        bit_depth=image.bit_depth,
    )

    return result_image
