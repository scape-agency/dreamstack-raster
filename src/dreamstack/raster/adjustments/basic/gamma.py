"""Gamma correction function."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dreamstack.raster.core.image import Image


def gamma(image: Image, gamma_value: float = 1.0) -> Image:
    """
    Apply gamma correction.

    Args:
        image: Input image
        gamma_value: Gamma value (0.1 to 10)

    Returns:
        Adjusted image
    """
    from dreamstack.raster.core.pixel import PixelData

    data = image.data.astype(np.float32)
    max_val = 255 if image.bit_depth.name == "UINT8" else 65535

    # Normalize
    normalized = data / max_val

    # Apply gamma
    result = np.power(normalized, 1 / gamma_value)

    # Scale back
    result = result * max_val
    result = np.clip(result, 0, max_val)

    result_image = image.copy()
    result_image._pixel_data = PixelData(
        data=result.astype(image.data.dtype),
        pixel_format=image.pixel_format,
        bit_depth=image.bit_depth,
    )

    return result_image
