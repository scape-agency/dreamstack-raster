# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Grain
=========================

Film grain effect implementation.

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


def grain(
    image: Image,
    intensity: float = 25,
    size: float = 1.0,
    roughness: float = 0.5,
) -> Image:
    """
    Add film grain effect.

    Args:
        image: Input image
        intensity: Grain intensity
        size: Grain size (1.0 = pixel-level)
        roughness: Grain roughness (0-1)

    Returns:
        Image with grain
    """
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.pixel import PixelData

    data = image.data.astype(np.float32)
    max_val = 255 if image.bit_depth.name == "UINT8" else 65535

    h, w = data.shape[:2]

    # Generate noise at potentially lower resolution
    if size > 1:
        small_h = max(1, int(h / size))
        small_w = max(1, int(w / size))
        noise_small = np.random.normal(0, intensity, (small_h, small_w))
        noise = cv2.resize(noise_small, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        noise = np.random.normal(0, intensity, (h, w))

    # Add roughness variation
    if roughness > 0:
        rough_noise = np.random.normal(0, intensity * roughness, (h, w))
        noise = noise * (1 - roughness) + rough_noise * roughness

    # Apply to all channels
    if data.ndim == 3:
        noise = np.repeat(noise[:, :, np.newaxis], data.shape[2], axis=2)

    result = data + noise
    result = np.clip(result, 0, max_val)

    result_image = image.copy()
    result_image._pixel_data = PixelData(  # pylint: disable=protected-access
        data=result.astype(image.data.dtype),
        pixel_format=image.pixel_format,
        bit_depth=image.bit_depth,
    )

    return result_image
