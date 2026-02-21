# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""Sepia tone effect function."""


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


def sepia(image: Image, intensity: float = 100) -> Image:
    """
    Apply sepia tone effect.

    Args:
        image: Input image
        intensity: Effect intensity (0-100)

    Returns:
        Sepia-toned image
    """
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.pixel import PixelData

    if image.channels < 3:
        return image.copy()

    data = image.data.astype(np.float32)
    max_val = 255 if image.bit_depth.name == "UINT8" else 65535

    normalized = data[:, :, :3] / max_val

    # Sepia matrix
    sepia_matrix = np.array(
        [[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]]
    )

    # Apply sepia
    sepia_result = np.zeros_like(normalized)
    for c in range(3):
        sepia_result[:, :, c] = (
            normalized[:, :, 0] * sepia_matrix[c, 0]
            + normalized[:, :, 1] * sepia_matrix[c, 1]
            + normalized[:, :, 2] * sepia_matrix[c, 2]
        )

    # Blend with original based on intensity
    blend = intensity / 100
    result = normalized * (1 - blend) + sepia_result * blend
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
