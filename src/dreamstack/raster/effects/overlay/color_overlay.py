# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Color Overlay
=============

Apply solid color overlay to images.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import cv2
import numpy as np

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel
    from numpy.typing import NDArray


def color_overlay(
    image: NDArray[np.uint8],
    color: tuple[int, int, int] = (255, 0, 0),
    *,
    opacity: float = 1.0,
    blend_mode: Literal["normal", "multiply", "screen", "overlay"] = "normal",
) -> NDArray[np.uint8]:
    """Apply a solid color overlay to an image.

    Fills the image content with a solid color.

    Args:
        image: Input image with alpha channel.
        color: Overlay color (RGB).
        opacity: Overlay opacity (0.0 to 1.0).
        blend_mode: Blend mode for overlay.

    Returns:
        Image with color overlay applied.

    Example:
        >>> result = color_overlay(image, (255, 0, 0), opacity=0.5)
    """
    # Ensure BGRA
    if image.ndim == 2:
        img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    elif image.shape[2] == 3:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    else:
        img = image.copy()

    result = img.astype(np.float32)
    alpha = img[:, :, 3].astype(np.float32) / 255.0

    overlay_color = np.array([color[2], color[1], color[0]], dtype=np.float32)

    for c in range(3):
        base = result[:, :, c] / 255.0
        blend = overlay_color[c] / 255.0

        if blend_mode == "normal":
            blended = blend
        elif blend_mode == "multiply":
            blended = base * blend
        elif blend_mode == "screen":
            blended = 1 - (1 - base) * (1 - blend)
        elif blend_mode == "overlay":
            blended = np.where(
                base < 0.5,
                2 * base * blend,
                1 - 2 * (1 - base) * (1 - blend),
            )
        else:
            blended = blend

        # Apply with opacity and alpha mask
        result[:, :, c] = (
            base * (1 - opacity * alpha) + blended * opacity * alpha
        ) * 255

    return np.clip(result, 0, 255).astype(np.uint8)
