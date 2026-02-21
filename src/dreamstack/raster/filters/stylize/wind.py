# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Wind
========================

Wind effect implementation.

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


def wind(
    image: Image,
    direction: str = "right",
    strength: int = 10,
    mode: str = "wind",
) -> Image:
    """
    Apply wind effect.

    Args:
        image: Input image
        direction: 'left' or 'right'
        strength: Wind strength
        mode: 'wind', 'blast', or 'stagger'

    Returns:
        Windswept image
    """
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.pixel import PixelData

    data = image.data.astype(np.float32)
    h, w = data.shape[:2]

    # Convert to grayscale for edge detection
    if data.ndim == 3 and data.shape[2] >= 3:
        gray = cv2.cvtColor(
            (
                data[:, :, :3].astype(np.uint8)
                if data.max() > 1
                else (data[:, :, :3] * 255).astype(np.uint8)
            ),
            cv2.COLOR_BGR2GRAY,
        )
    else:
        gray = data[:, :, 0] if data.ndim == 3 else data
        if gray.max() <= 1:
            gray = (gray * 255).astype(np.uint8)

    # Detect edges
    edges = cv2.Canny(gray, 100, 200)

    result = data.copy()

    for y in range(h):
        for x in range(w):
            if edges[y, x] > 0:
                # Apply wind streak
                if mode == "wind":
                    streak_length = np.random.randint(1, strength + 1)
                elif mode == "blast":
                    streak_length = strength
                else:  # stagger
                    streak_length = np.random.choice(
                        [1, strength // 2, strength]
                    )

                if direction == "right":
                    for i in range(streak_length):
                        nx = x + i
                        if nx < w:
                            # Fade the streak
                            alpha = 1 - i / streak_length
                            if data.ndim == 3:
                                result[y, nx] = (
                                    alpha * data[y, x]
                                    + (1 - alpha) * result[y, nx]
                                )
                            else:
                                result[y, nx] = (
                                    alpha * data[y, x]
                                    + (1 - alpha) * result[y, nx]
                                )
                else:  # left
                    for i in range(streak_length):
                        nx = x - i
                        if nx >= 0:
                            alpha = 1 - i / streak_length
                            if data.ndim == 3:
                                result[y, nx] = (
                                    alpha * data[y, x]
                                    + (1 - alpha) * result[y, nx]
                                )
                            else:
                                result[y, nx] = (
                                    alpha * data[y, x]
                                    + (1 - alpha) * result[y, nx]
                                )

    result_image = image.copy()
    result_image._pixel_data = PixelData(  # pylint: disable=protected-access
        data=result.astype(image.data.dtype),
        pixel_format=image.pixel_format,
        bit_depth=image.bit_depth,
    )

    return result_image
