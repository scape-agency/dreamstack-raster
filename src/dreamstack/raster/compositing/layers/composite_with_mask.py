"""Mask-based compositing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def composite_with_mask(
    foreground: NDArray[np.uint8],
    background: NDArray[np.uint8],
    mask: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    """Composite foreground onto background using a mask.

    Args:
        foreground: Foreground image (RGB/RGBA).
        background: Background image (RGB/RGBA).
        mask: Grayscale mask (white = foreground, black = background).

    Returns:
        Composited image.

    Example:
        >>> result = composite_with_mask(person, scene, person_mask)
    """
    import cv2

    # Ensure same size
    h, w = background.shape[:2]
    if foreground.shape[:2] != (h, w):
        foreground = cv2.resize(foreground, (w, h))
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h))

    # Ensure mask is single channel
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Normalize mask to 0-1
    alpha = mask.astype(np.float32) / 255.0

    # Ensure same channel count
    if foreground.ndim == 2:
        foreground = cv2.cvtColor(foreground, cv2.COLOR_GRAY2BGR)
    if background.ndim == 2:
        background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)

    # Handle alpha channels
    if foreground.shape[2] == 4:
        foreground = foreground[:, :, :3]
    if background.shape[2] == 4:
        background = background[:, :, :3]

    # Expand alpha for broadcasting
    alpha = alpha[:, :, np.newaxis]

    # Blend
    result = foreground.astype(np.float32) * alpha + background.astype(np.float32) * (
        1 - alpha
    )

    return result.astype(np.uint8)
