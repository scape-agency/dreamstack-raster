# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Drop Shadow
===========

Create drop shadow effects.

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
    from numpy.typing import NDArray


def drop_shadow(
    image: NDArray[np.uint8],
    *,
    offset: tuple[int, int] = (5, 5),
    blur: float = 10.0,
    color: tuple[int, int, int] = (0, 0, 0),
    opacity: float = 0.5,
    spread: int = 0,
) -> NDArray[np.uint8]:
    """Add a drop shadow effect to an image.

    Creates a soft shadow behind the image content.

    Args:
        image: Input image with alpha channel.
        offset: Shadow offset (x, y) in pixels.
        blur: Shadow blur radius.
        color: Shadow color (RGB).
        opacity: Shadow opacity (0.0 to 1.0).
        spread: Shadow spread in pixels (expands shadow).

    Returns:
        Image with drop shadow applied.

    Example:
        >>> result = drop_shadow(image, offset=(10, 10), blur=15)
    """
    # Ensure BGRA
    if image.ndim == 2:
        img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    elif image.shape[2] == 3:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    else:
        img = image

    h, w = img.shape[:2]

    # Get alpha mask
    alpha = img[:, :, 3].astype(np.float32) / 255.0

    # Expand mask for offset
    ox, oy = offset
    pad = max(abs(ox), abs(oy)) + int(blur * 3) + spread

    # Create padded shadow
    shadow_h = h + 2 * pad
    shadow_w = w + 2 * pad
    shadow = np.zeros((shadow_h, shadow_w), dtype=np.float32)

    # Place alpha in offset position
    sy = pad + oy
    sx = pad + ox
    shadow[sy : sy + h, sx : sx + w] = alpha

    # Apply spread
    if spread > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (spread * 2 + 1, spread * 2 + 1)
        )
        shadow = cv2.dilate(shadow, kernel)

    # Apply blur
    if blur > 0:
        shadow = cv2.GaussianBlur(shadow, (0, 0), blur)

    # Apply opacity
    shadow = shadow * opacity

    # Create shadow layer
    shadow_layer = np.zeros((shadow_h, shadow_w, 4), dtype=np.float32)
    shadow_layer[:, :, 0] = color[2]  # B
    shadow_layer[:, :, 1] = color[1]  # G
    shadow_layer[:, :, 2] = color[0]  # R
    shadow_layer[:, :, 3] = shadow * 255

    # Create result with padding
    result = np.zeros((shadow_h, shadow_w, 4), dtype=np.float32)

    # Composite shadow first
    result[:, :, :3] = shadow_layer[:, :, :3]
    result[:, :, 3] = shadow_layer[:, :, 3]

    # Composite original image on top
    orig_y = pad
    orig_x = pad

    img_f = img.astype(np.float32)
    img_alpha = img_f[:, :, 3:4] / 255.0
    bg_alpha = result[orig_y : orig_y + h, orig_x : orig_x + w, 3:4] / 255.0

    # Over compositing
    out_alpha = img_alpha + bg_alpha * (1 - img_alpha)
    out_alpha_safe = np.maximum(out_alpha, 1e-6)

    out_rgb = (
        img_f[:, :, :3] * img_alpha
        + result[orig_y : orig_y + h, orig_x : orig_x + w, :3]
        * bg_alpha
        * (1 - img_alpha)
    ) / out_alpha_safe

    result[orig_y : orig_y + h, orig_x : orig_x + w, :3] = out_rgb
    result[orig_y : orig_y + h, orig_x : orig_x + w, 3:4] = out_alpha * 255

    # Crop back to original size with some padding for shadow
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result
