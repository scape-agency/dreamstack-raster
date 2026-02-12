"""
Pattern Overlay
===============

Apply pattern overlay to images.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def pattern_overlay(
    image: NDArray[np.uint8],
    pattern: NDArray[np.uint8],
    *,
    opacity: float = 1.0,
    scale: float = 1.0,
    blend_mode: Literal["normal", "multiply", "screen", "overlay"] = "normal",
) -> NDArray[np.uint8]:
    """Apply a pattern overlay to an image.

    Tiles a pattern across the image content.

    Args:
        image: Input image with alpha channel.
        pattern: Pattern image to tile.
        opacity: Overlay opacity (0.0 to 1.0).
        scale: Pattern scale factor.
        blend_mode: Blend mode for overlay.

    Returns:
        Image with pattern overlay applied.

    Example:
        >>> result = pattern_overlay(image, checkerboard, opacity=0.5)
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

    # Scale pattern
    pat = pattern
    if scale != 1.0:
        pat_h, pat_w = pat.shape[:2]
        new_h = max(1, int(pat_h * scale))
        new_w = max(1, int(pat_w * scale))
        pat = cv2.resize(pat, (new_w, new_h))

    # Ensure pattern is at least 3 channels
    if pat.ndim == 2:
        pat = cv2.cvtColor(pat, cv2.COLOR_GRAY2BGR)
    elif pat.shape[2] == 4:
        pat = pat[:, :, :3]

    # Tile pattern
    pat_h, pat_w = pat.shape[:2]
    tiled = np.zeros((h, w, 3), dtype=np.uint8)

    for y in range(0, h, pat_h):
        for x in range(0, w, pat_w):
            y_end = min(y + pat_h, h)
            x_end = min(x + pat_w, w)
            tiled[y:y_end, x:x_end] = pat[: y_end - y, : x_end - x]

    tiled_f = tiled.astype(np.float32) / 255.0

    for c in range(3):
        base = result[:, :, c] / 255.0
        pat_c = tiled_f[:, :, c]

        if blend_mode == "normal":
            blended = pat_c
        elif blend_mode == "multiply":
            blended = base * pat_c
        elif blend_mode == "screen":
            blended = 1 - (1 - base) * (1 - pat_c)
        elif blend_mode == "overlay":
            blended = np.where(
                base < 0.5,
                2 * base * pat_c,
                1 - 2 * (1 - base) * (1 - pat_c),
            )
        else:
            blended = pat_c

        result[:, :, c] = (
            base * (1 - opacity * alpha) + blended * opacity * alpha
        ) * 255

    return np.clip(result, 0, 255).astype(np.uint8)
