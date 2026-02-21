# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Smart Sharpen
=================================

Smart sharpening with blur type detection filter implementation.

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


def smart_sharpen(
    image: Image,
    amount: float = 100,
    radius: float = 1.0,
    reduce_noise: float = 10,
    remove: str = "gaussian",
) -> Image:
    """
    Apply smart sharpening with blur type detection.

    Args:
        image: Input image
        amount: Sharpening amount (0-500%)
        radius: Blur radius
        reduce_noise: Noise reduction amount
        remove: Blur type to remove ('gaussian', 'lens', 'motion')

    Returns:
        Sharpened image
    """
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.pixel import PixelData

    data = image.data.astype(np.float32)
    max_val = 255 if image.bit_depth.name == "UINT8" else 65535

    ksize = int(radius * 2) | 1
    ksize = max(ksize, 3)

    if remove == "gaussian":
        # Standard Gaussian blur for deconvolution
        if data.ndim == 3:
            blurred = np.zeros_like(data)
            for i in range(data.shape[2]):
                blurred[:, :, i] = cv2.GaussianBlur(
                    data[:, :, i], (ksize, ksize), radius / 3
                )
        else:
            blurred = cv2.GaussianBlur(data, (ksize, ksize), radius / 3)

    elif remove == "lens":
        # Disk blur approximation
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (ksize, ksize)
        ).astype(np.float32)
        kernel /= kernel.sum()

        if data.ndim == 3:
            blurred = np.zeros_like(data)
            for i in range(data.shape[2]):
                blurred[:, :, i] = cv2.filter2D(data[:, :, i], -1, kernel)
        else:
            blurred = cv2.filter2D(data, -1, kernel)

    elif remove == "motion":
        # Motion blur kernel
        kernel = np.zeros((ksize, ksize), dtype=np.float32)
        center = ksize // 2
        kernel[center, :] = 1
        kernel /= kernel.sum()

        if data.ndim == 3:
            blurred = np.zeros_like(data)
            for i in range(data.shape[2]):
                blurred[:, :, i] = cv2.filter2D(data[:, :, i], -1, kernel)
        else:
            blurred = cv2.filter2D(data, -1, kernel)
    else:
        raise ValueError(f"Unknown blur type: {remove}")

    # Calculate mask
    mask = data - blurred

    # Reduce noise in mask
    if reduce_noise > 0:
        noise_ksize = int(reduce_noise) | 1
        if noise_ksize >= 3:
            if mask.ndim == 3:
                for i in range(mask.shape[2]):
                    mask[:, :, i] = cv2.GaussianBlur(
                        mask[:, :, i], (noise_ksize, noise_ksize), 0
                    )
            else:
                mask = cv2.GaussianBlur(mask, (noise_ksize, noise_ksize), 0)

    # Apply sharpening
    amount_factor = amount / 100.0
    result = data + mask * amount_factor

    result = np.clip(result, 0, max_val)

    result_image = image.copy()
    result_image._pixel_data = PixelData(  # pylint: disable=protected-access
        data=result.astype(image.data.dtype),
        pixel_format=image.pixel_format,
        bit_depth=image.bit_depth,
    )

    return result_image
