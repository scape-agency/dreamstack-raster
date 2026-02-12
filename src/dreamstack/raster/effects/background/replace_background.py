"""
Replace Background
==================

Replace background of RGBA image with another image.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def replace_background(
    rgba_image: NDArray[np.uint8],
    background: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    """Replace background of RGBA image with another image.

    Composites the RGBA foreground onto the provided background image.
    Background will be resized to match the foreground dimensions.

    Args:
        rgba_image: RGBA image with alpha channel (4 channels).
        background: Background image (RGB, 3 channels).

    Returns:
        RGB image with replaced background.

    Example:
        >>> from dreamstack.raster.effects.background import (
        ...     remove_background,
        ...     replace_background,
        ... )
        >>> rgba = remove_background(image)
        >>> result = replace_background(rgba, new_bg)
    """
    import cv2

    if rgba_image.ndim != 3 or rgba_image.shape[2] != 4:
        raise ValueError("Expected RGBA image with 4 channels")

    h, w = rgba_image.shape[:2]

    # Resize background if needed
    if background.shape[:2] != (h, w):
        background = cv2.resize(background, (w, h), interpolation=cv2.INTER_LINEAR)

    # Ensure background is 3 channels
    if background.ndim == 2:
        background = cv2.cvtColor(background, cv2.COLOR_GRAY2RGB)
    elif background.shape[2] == 4:
        background = background[:, :, :3]

    # Extract alpha and normalize
    alpha = rgba_image[:, :, 3:4].astype(np.float32) / 255.0
    fg = rgba_image[:, :, :3].astype(np.float32)
    bg = background.astype(np.float32)

    # Alpha blend
    result = fg * alpha + bg * (1 - alpha)

    return result.astype(np.uint8)
