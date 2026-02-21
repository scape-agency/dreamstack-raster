# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Glitch
==========================

Digital glitch effect implementation.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.image import Image


def glitch(image: Image, amount: float = 10, seed: int | None = None) -> Image:
    """
    Apply digital glitch effect.

    Args:
        image: Input image
        amount: Glitch intensity
        seed: Random seed for reproducibility

    Returns:
        Glitched image
    """
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.pixel import PixelData

    if seed is not None:
        np.random.seed(seed)

    data = image.data.copy()
    h, w = data.shape[:2]

    # Random horizontal shifts
    num_shifts = int(amount)
    for _ in range(num_shifts):
        y = np.random.randint(0, h)
        height = np.random.randint(1, max(2, h // 10))
        shift = np.random.randint(
            -int(w * amount / 100), int(w * amount / 100)
        )

        y2 = min(y + height, h)
        data[y:y2] = np.roll(data[y:y2], shift, axis=1)

    # Color channel separation
    if data.ndim == 3 and data.shape[2] >= 3:
        shift_r = np.random.randint(-int(amount), int(amount) + 1)
        shift_b = np.random.randint(-int(amount), int(amount) + 1)

        data[:, :, 0] = np.roll(data[:, :, 0], shift_r, axis=1)
        data[:, :, 2] = np.roll(data[:, :, 2], shift_b, axis=1)

    result_image = image.copy()
    result_image._pixel_data = PixelData(  # pylint: disable=protected-access
        data=data, pixel_format=image.pixel_format, bit_depth=image.bit_depth
    )

    return result_image
