# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Outer Glow
==========

Create outer glow effects.

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


def outer_glow(
    image: NDArray[np.uint8],
    *,
    color: tuple[int, int, int] = (255, 255, 0),
    blur: float = 10.0,
    opacity: float = 0.75,
    spread: int = 0,
) -> NDArray[np.uint8]:
    """Add an outer glow effect to an image.

    Creates a glowing halo around the image content.

    Args:
        image: Input image with alpha channel.
        color: Glow color (RGB).
        blur: Glow blur radius.
        opacity: Glow opacity (0.0 to 1.0).
        spread: Glow spread in pixels.

    Returns:
        Image with outer glow applied.

    Example:
        >>> result = outer_glow(image, color=(255, 200, 0), blur=20)
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

    # Expand for glow
    pad = int(blur * 3) + spread
    glow_h = h + 2 * pad
    glow_w = w + 2 * pad

    glow = np.zeros((glow_h, glow_w), dtype=np.float32)
    glow[pad : pad + h, pad : pad + w] = alpha

    # Apply spread (dilation)
    if spread > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (spread * 2 + 1, spread * 2 + 1)
        )
        glow = cv2.dilate(glow, kernel)

    # Apply blur
    if blur > 0:
        glow = cv2.GaussianBlur(glow, (0, 0), blur)

    # Apply opacity
    glow = glow * opacity

    # Create glow layer
    glow_layer = np.zeros((glow_h, glow_w, 4), dtype=np.float32)
    glow_layer[:, :, 0] = color[2]  # B
    glow_layer[:, :, 1] = color[1]  # G
    glow_layer[:, :, 2] = color[0]  # R
    glow_layer[:, :, 3] = glow * 255

    # Create result
    result = glow_layer.copy()

    # Composite original on top
    img_f = img.astype(np.float32)
    img_alpha = img_f[:, :, 3:4] / 255.0
    bg_alpha = result[pad : pad + h, pad : pad + w, 3:4] / 255.0

    out_alpha = img_alpha + bg_alpha * (1 - img_alpha)
    out_alpha_safe = np.maximum(out_alpha, 1e-6)

    out_rgb = (
        img_f[:, :, :3] * img_alpha
        + result[pad : pad + h, pad : pad + w, :3] * bg_alpha * (1 - img_alpha)
    ) / out_alpha_safe

    result[pad : pad + h, pad : pad + w, :3] = out_rgb
    result[pad : pad + h, pad : pad + w, 3:4] = out_alpha * 255

    return np.clip(result, 0, 255).astype(np.uint8)
