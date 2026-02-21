# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Gradient Overlay
================

Apply gradient overlay to images.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import cv2
import numpy as np

# =============================================================================
# Type Checking Imports
# =============================================================================

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel
    from numpy.typing import NDArray


def gradient_overlay(
    image: NDArray[np.uint8],
    start_color: tuple[int, int, int] = (0, 0, 0),
    end_color: tuple[int, int, int] = (255, 255, 255),
    *,
    angle: float = 90.0,
    opacity: float = 1.0,
    blend_mode: Literal["normal", "multiply", "screen", "overlay"] = "normal",
    reverse: bool = False,
) -> NDArray[np.uint8]:
    """Apply a gradient overlay to an image.

    Creates a gradient that fills the image content.

    Args:
        image: Input image with alpha channel.
        start_color: Starting color (RGB).
        end_color: Ending color (RGB).
        angle: Gradient angle in degrees (0 = left, 90 = top).
        opacity: Overlay opacity (0.0 to 1.0).
        blend_mode: Blend mode for overlay.
        reverse: If True, reverse gradient direction.

    Returns:
        Image with gradient overlay applied.

    Example:
        >>> result = gradient_overlay(image, (255, 0, 0), (0, 0, 255), angle=45)
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
    alpha = img[:, :, 3].astype(np.float32) / 255.0

    # Create gradient coordinates
    y_coords, x_coords = np.mgrid[0:h, 0:w]

    # Calculate gradient direction
    angle_rad = np.radians(angle)
    dx = np.cos(angle_rad)
    dy = -np.sin(angle_rad)  # Negative because y increases downward

    # Project coordinates
    cx, cy = w / 2, h / 2
    gradient = (x_coords - cx) * dx + (y_coords - cy) * dy

    # Normalize to [0, 1]
    g_min, g_max = gradient.min(), gradient.max()
    if g_max > g_min:
        t = (gradient - g_min) / (g_max - g_min)
    else:
        t = np.zeros_like(gradient)

    if reverse:
        t = 1 - t

    # Interpolate colors
    start_bgr = np.array(
        [start_color[2], start_color[1], start_color[0]], dtype=np.float32
    )
    end_bgr = np.array(
        [end_color[2], end_color[1], end_color[0]], dtype=np.float32
    )

    for c in range(3):
        base = result[:, :, c] / 255.0
        grad_color = (start_bgr[c] * (1 - t) + end_bgr[c] * t) / 255.0

        if blend_mode == "normal":
            blended = grad_color
        elif blend_mode == "multiply":
            blended = base * grad_color
        elif blend_mode == "screen":
            blended = 1 - (1 - base) * (1 - grad_color)
        elif blend_mode == "overlay":
            blended = np.where(
                base < 0.5,
                2 * base * grad_color,
                1 - 2 * (1 - base) * (1 - grad_color),
            )
        else:
            blended = grad_color

        result[:, :, c] = (
            base * (1 - opacity * alpha) + blended * opacity * alpha
        ) * 255

    return np.clip(result, 0, 255).astype(np.uint8)
