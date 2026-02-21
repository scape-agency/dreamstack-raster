# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Black and white conversion function."""


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

from dreamstack.raster.adjustments.black_white._color_utils import _hue_to_rgb


def black_white(
    image: Image,
    reds: float = 40,
    yellows: float = 60,
    greens: float = 40,
    cyans: float = 60,
    blues: float = 20,
    magentas: float = 80,
    tint_hue: float = 0,
    tint_saturation: float = 0,
) -> Image:
    """
    Convert to black and white with color channel control.

    Args:
        image: Input image
        reds: Red channel contribution (-200 to 300)
        yellows: Yellow channel contribution (-200 to 300)
        greens: Green channel contribution (-200 to 300)
        cyans: Cyan channel contribution (-200 to 300)
        blues: Blue channel contribution (-200 to 300)
        magentas: Magenta channel contribution (-200 to 300)
        tint_hue: Tint hue for sepia/toning (0-360)
        tint_saturation: Tint saturation (0-100)

    Returns:
        Black and white image
    """
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.pixel import PixelData

    if image.channels < 3:
        return image.copy()

    data = image.data.astype(np.float32)
    max_val = 255 if image.bit_depth.name == "UINT8" else 65535

    normalized = data[:, :, :3] / max_val
    r, g, b = normalized[:, :, 0], normalized[:, :, 1], normalized[:, :, 2]

    # Calculate color contributions
    # Each slider affects specific hue ranges

    # Red contribution (red channel minus green and blue influence)
    red_contrib = r * reds / 100

    # Yellow contribution (red + green)
    yellow = np.minimum(r, g)
    yellow_contrib = yellow * yellows / 100

    # Green contribution
    green_contrib = g * greens / 100

    # Cyan contribution (green + blue)
    cyan = np.minimum(g, b)
    cyan_contrib = cyan * cyans / 100

    # Blue contribution
    blue_contrib = b * blues / 100

    # Magenta contribution (red + blue)
    magenta = np.minimum(r, b)
    magenta_contrib = magenta * magentas / 100

    # Combine with weighting
    # Base grayscale
    gray = (
        red_contrib
        + yellow_contrib
        + green_contrib
        + cyan_contrib
        + blue_contrib
        + magenta_contrib
    ) / 3

    # Normalize
    gray = np.clip(gray, 0, 1)

    # Apply tint
    if tint_saturation > 0:
        # Convert tint hue to RGB
        tint_rgb = _hue_to_rgb(tint_hue)

        sat = tint_saturation / 100

        result = np.zeros((*gray.shape, 3))
        for i in range(3):
            # Mix gray with tint color
            result[:, :, i] = gray * (1 - sat) + gray * tint_rgb[i] * sat
    else:
        result = np.stack([gray, gray, gray], axis=-1)

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
