# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Inner Shadow
============

Create inner shadow effects.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def inner_shadow(
    image: NDArray[np.uint8],
    *,
    offset: tuple[int, int] = (3, 3),
    blur: float = 5.0,
    color: tuple[int, int, int] = (0, 0, 0),
    opacity: float = 0.5,
    choke: int = 0,
) -> NDArray[np.uint8]:
    """Add an inner shadow effect to an image.

    Creates a shadow inside the image content edges.

    Args:
        image: Input image with alpha channel.
        offset: Shadow offset (x, y) in pixels.
        blur: Shadow blur radius.
        color: Shadow color (RGB).
        opacity: Shadow opacity (0.0 to 1.0).
        choke: Shadow choke in pixels (contracts shadow).

    Returns:
        Image with inner shadow applied.

    Example:
        >>> result = inner_shadow(image, offset=(2, 2), blur=5)
    """
    # Ensure BGRA
    if image.ndim == 2:
        img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    elif image.shape[2] == 3:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    else:
        img = image.copy()

    h, w = img.shape[:2]
    result = img.astype(np.float32)

    # Get alpha mask
    alpha = img[:, :, 3].astype(np.float32) / 255.0

    # Create inverted and offset alpha
    ox, oy = offset
    inverted = np.zeros_like(alpha)

    # Offset the inverted alpha
    src_y1 = max(0, -oy)
    src_y2 = min(h, h - oy)
    src_x1 = max(0, -ox)
    src_x2 = min(w, w - ox)

    dst_y1 = max(0, oy)
    dst_y2 = min(h, h + oy)
    dst_x1 = max(0, ox)
    dst_x2 = min(w, w + ox)

    inverted[dst_y1:dst_y2, dst_x1:dst_x2] = (
        1.0 - alpha[src_y1:src_y2, src_x1:src_x2]
    )

    # Apply choke
    if choke > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (choke * 2 + 1, choke * 2 + 1)
        )
        inverted = cv2.erode(inverted, kernel)

    # Apply blur
    if blur > 0:
        inverted = cv2.GaussianBlur(inverted, (0, 0), blur)

    # Mask to only show inside content
    shadow = inverted * alpha * opacity

    # Apply shadow color
    shadow_color = np.array([color[2], color[1], color[0]], dtype=np.float32)

    # Blend shadow with image
    for c in range(3):
        result[:, :, c] = (
            result[:, :, c] * (1 - shadow) + shadow_color[c] * shadow
        )

    return np.clip(result, 0, 255).astype(np.uint8)
