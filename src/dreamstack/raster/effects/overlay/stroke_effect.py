# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Stroke Effect
=============

Apply stroke/outline effect to images.

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


def stroke_effect(
    image: NDArray[np.uint8],
    color: tuple[int, int, int] = (0, 0, 0),
    *,
    size: int = 3,
    position: Literal["outside", "inside", "center"] = "outside",
    opacity: float = 1.0,
) -> NDArray[np.uint8]:
    """Add a stroke/outline effect to an image.

    Creates an outline around the image content.

    Args:
        image: Input image with alpha channel.
        color: Stroke color (RGB).
        size: Stroke size in pixels.
        position: Stroke position relative to edge.
        opacity: Stroke opacity (0.0 to 1.0).

    Returns:
        Image with stroke applied.

    Example:
        >>> result = stroke_effect(image, (255, 255, 255), size=5)
    """
    # Ensure BGRA
    if image.ndim == 2:
        img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    elif image.shape[2] == 3:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    else:
        img = image

    h, w = img.shape[:2]
    alpha = img[:, :, 3].astype(np.float32) / 255.0

    # Create stroke mask based on position
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (size * 2 + 1, size * 2 + 1)
    )
    pad = 0  # Will be set for outside/center positions

    if position == "outside":
        dilated = cv2.dilate(alpha, kernel)
        stroke_mask = dilated - alpha

        # Expand canvas for outside stroke
        pad = size
        padded_h = h + 2 * pad
        padded_w = w + 2 * pad
        result = np.zeros((padded_h, padded_w, 4), dtype=np.float32)

        # Place stroke
        dilated_padded = np.zeros((padded_h, padded_w), dtype=np.float32)
        dilated_padded[pad : pad + h, pad : pad + w] = dilated
        stroke_mask_padded = dilated_padded.copy()
        stroke_mask_padded[pad : pad + h, pad : pad + w] -= alpha

        stroke_mask = stroke_mask_padded

    elif position == "inside":
        eroded = cv2.erode(alpha, kernel)
        stroke_mask = alpha - eroded
        result = img.astype(np.float32)

    else:  # center
        dilated = cv2.dilate(alpha, kernel)
        eroded = cv2.erode(alpha, kernel)
        stroke_mask = dilated - eroded

        # Expand canvas slightly
        pad = size // 2 + 1
        padded_h = h + 2 * pad
        padded_w = w + 2 * pad
        result = np.zeros((padded_h, padded_w, 4), dtype=np.float32)

        dilated_padded = np.zeros((padded_h, padded_w), dtype=np.float32)
        dilated_padded[pad : pad + h, pad : pad + w] = dilated
        eroded_padded = np.zeros((padded_h, padded_w), dtype=np.float32)
        eroded_padded[pad : pad + h, pad : pad + w] = eroded

        stroke_mask = dilated_padded - eroded_padded

    stroke_mask = stroke_mask * opacity

    # Apply stroke color
    stroke_bgr = np.array([color[2], color[1], color[0]], dtype=np.float32)

    if position == "inside":
        for c in range(3):
            base = result[:, :, c] / 255.0
            result[:, :, c] = (
                base * (1 - stroke_mask) + stroke_bgr[c] * stroke_mask
            ) * 255
    else:
        # For outside/center, composite stroke then image
        for c in range(3):
            result[:, :, c] = stroke_bgr[c] * stroke_mask
        result[:, :, 3] = stroke_mask * 255

        # Composite original on top
        if position == "outside":
            img_f = img.astype(np.float32)
            img_alpha = img_f[:, :, 3:4] / 255.0
            bg_alpha = result[pad : pad + h, pad : pad + w, 3:4] / 255.0

            out_alpha = img_alpha + bg_alpha * (1 - img_alpha)
            out_alpha_safe = np.maximum(out_alpha, 1e-6)

            out_rgb = (
                img_f[:, :, :3] * img_alpha
                + result[pad : pad + h, pad : pad + w, :3]
                * bg_alpha
                * (1 - img_alpha)
            ) / out_alpha_safe

            result[pad : pad + h, pad : pad + w, :3] = out_rgb
            result[pad : pad + h, pad : pad + w, 3:4] = out_alpha * 255
        else:
            # Center stroke
            img_f = img.astype(np.float32)
            img_alpha = img_f[:, :, 3:4] / 255.0
            bg_alpha = result[pad : pad + h, pad : pad + w, 3:4] / 255.0

            out_alpha = img_alpha + bg_alpha * (1 - img_alpha)
            out_alpha_safe = np.maximum(out_alpha, 1e-6)

            out_rgb = (
                img_f[:, :, :3] * img_alpha
                + result[pad : pad + h, pad : pad + w, :3]
                * bg_alpha
                * (1 - img_alpha)
            ) / out_alpha_safe

            result[pad : pad + h, pad : pad + w, :3] = out_rgb
            result[pad : pad + h, pad : pad + w, 3:4] = out_alpha * 255

    return np.clip(result, 0, 255).astype(np.uint8)
