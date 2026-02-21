# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Match color function."""


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


def match_color(
    image: Image,
    reference: Image,
    luminance: float = 100,
    color_intensity: float = 100,
    fade: float = 0,
) -> Image:
    """
    Match colors to a reference image.

    Args:
        image: Input image
        reference: Reference image for color matching
        luminance: Luminance matching strength (0-200)
        color_intensity: Color matching strength (0-200)
        fade: Fade towards neutral (0-100)

    Returns:
        Color-matched image
    """
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.pixel import PixelData

    if image.channels < 3 or reference.channels < 3:
        return image.copy()

    data = image.data.astype(np.float32)
    ref_data = reference.data.astype(np.float32)
    max_val = 255 if image.bit_depth.name == "UINT8" else 65535
    ref_max = 255 if reference.bit_depth.name == "UINT8" else 65535

    # Normalize
    normalized = data[:, :, :3] / max_val
    ref_normalized = ref_data[:, :, :3] / ref_max

    # Calculate statistics
    src_mean = np.mean(normalized, axis=(0, 1))
    src_std = np.std(normalized, axis=(0, 1))

    ref_mean = np.mean(ref_normalized, axis=(0, 1))
    ref_std = np.std(ref_normalized, axis=(0, 1))

    # Apply color transfer
    result = normalized.copy()

    for i in range(3):
        if src_std[i] > 0:
            # Normalize to mean 0, std 1
            result[:, :, i] = (result[:, :, i] - src_mean[i]) / src_std[i]

            # Scale to reference statistics
            scale = (luminance / 100) * (color_intensity / 100)
            result[:, :, i] = result[:, :, i] * (
                src_std[i] * (1 - scale) + ref_std[i] * scale
            )
            result[:, :, i] = (
                result[:, :, i]
                + src_mean[i] * (1 - scale)
                + ref_mean[i] * scale
            )

    # Apply fade
    if fade > 0:
        gray = (
            0.299 * result[:, :, 0]
            + 0.587 * result[:, :, 1]
            + 0.114 * result[:, :, 2]
        )
        fade_factor = fade / 100
        result = (
            result * (1 - fade_factor) + gray[:, :, np.newaxis] * fade_factor
        )

    result = np.clip(result, 0, 1)

    final = data.copy()
    final[:, :, :3] = result * max_val

    result_image = image.copy()
    result_image._pixel_data = PixelData(  # pylint: disable=protected-access
        data=final.astype(image.data.dtype),
        pixel_format=image.pixel_format,
        bit_depth=image.bit_depth,
    )

    return result_image
