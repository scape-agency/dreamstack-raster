# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""Exposure adjustment function."""


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


def exposure(
    image: Image,
    exposure_amount: float = 0,
    offset: float = 0,
    gamma_correction: float = 1.0,
) -> Image:
    """
    Adjust exposure like a camera.

    Args:
        image: Input image
        exposure_amount: Exposure in stops (-20 to 20)
        offset: Shadow offset (-0.5 to 0.5)
        gamma_correction: Gamma (0.01 to 9.99)

    Returns:
        Adjusted image
    """
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.pixel import PixelData

    data = image.data.astype(np.float32)
    max_val = 255 if image.bit_depth.name == "UINT8" else 65535

    # Normalize to 0-1
    normalized = data / max_val

    # Apply exposure (in stops)
    exposure_factor = 2**exposure_amount
    result = normalized * exposure_factor

    # Apply offset
    result = result + offset

    # Apply gamma
    result = np.power(np.clip(result, 0, None), 1 / gamma_correction)

    # Scale back
    result = result * max_val
    result = np.clip(result, 0, max_val)

    result_image = image.copy()
    result_image._pixel_data = PixelData(  # pylint: disable=protected-access
        data=result.astype(image.data.dtype),
        pixel_format=image.pixel_format,
        bit_depth=image.bit_depth,
    )

    return result_image
