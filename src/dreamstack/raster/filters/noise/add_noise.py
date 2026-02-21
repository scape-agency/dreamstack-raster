# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Add Noise
=============================

Noise addition filter implementation.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

# =============================================================================
# Type Checking Imports
# =============================================================================

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.image import Image


def add_noise(
    image: Image,
    amount: float = 10,
    noise_type: str = "gaussian",
    monochromatic: bool = False,
) -> Image:
    """
    Add noise to image.

    Args:
        image: Input image
        amount: Noise amount (standard deviation for gaussian)
        noise_type: Type of noise ('gaussian', 'uniform', 'salt_pepper', 'poisson')
        monochromatic: Use same noise for all channels

    Returns:
        Noisy image
    """
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.pixel import PixelData

    data = image.data.astype(np.float32)
    max_val = 255 if image.bit_depth.name == "UINT8" else 65535

    h, w = data.shape[:2]
    channels = data.shape[2] if data.ndim == 3 else 1

    if noise_type == "gaussian":
        if monochromatic:
            noise = np.random.normal(0, amount, (h, w))
            if channels > 1:
                noise = np.repeat(noise[:, :, np.newaxis], channels, axis=2)
        else:
            noise = np.random.normal(0, amount, data.shape)

        result = data + noise

    elif noise_type == "uniform":
        if monochromatic:
            noise = np.random.uniform(-amount, amount, (h, w))
            if channels > 1:
                noise = np.repeat(noise[:, :, np.newaxis], channels, axis=2)
        else:
            noise = np.random.uniform(-amount, amount, data.shape)

        result = data + noise

    elif noise_type == "salt_pepper":
        result = data.copy()
        prob = amount / 100.0

        # Salt
        salt_mask = np.random.random(data.shape[:2]) < prob / 2
        if data.ndim == 3:
            for i in range(channels):
                result[:, :, i][salt_mask] = max_val
        else:
            result[salt_mask] = max_val

        # Pepper
        pepper_mask = np.random.random(data.shape[:2]) < prob / 2
        if data.ndim == 3:
            for i in range(channels):
                result[:, :, i][pepper_mask] = 0
        else:
            result[pepper_mask] = 0

    elif noise_type == "poisson":
        # Scale for Poisson distribution
        scale = max_val / amount if amount > 0 else 1
        noisy = np.random.poisson(data / scale) * scale
        result = noisy

    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    result = np.clip(result, 0, max_val)

    result_image = image.copy()
    result_image._pixel_data = PixelData(  # pylint: disable=protected-access
        data=result.astype(image.data.dtype),
        pixel_format=image.pixel_format,
        bit_depth=image.bit_depth,
    )

    return result_image
