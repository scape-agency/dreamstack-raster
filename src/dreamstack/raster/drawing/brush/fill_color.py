"""
Fill Color
==========

Flood fill and bucket fill operations.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def fill_color(
    image: NDArray[np.uint8],
    x: int,
    y: int,
    color: tuple[int, int, int] | tuple[int, int, int, int],
    *,
    tolerance: int = 32,
    contiguous: bool = True,
) -> NDArray[np.uint8]:
    """Fill an area with color.

    Uses flood fill to replace colors within tolerance.

    Args:
        image: Image to fill.
        x: Seed x coordinate.
        y: Seed y coordinate.
        color: Fill color (RGB or RGBA).
        tolerance: Color tolerance for fill expansion (0-255).
        contiguous: If True, only fill connected pixels.

    Returns:
        Image with fill applied.

    Example:
        >>> # Fill red where clicked
        >>> result = fill_color(image, 100, 100, (255, 0, 0))
        >>>
        >>> # Fill with tolerance for similar colors
        >>> result = fill_color(image, 50, 50, (0, 255, 0), tolerance=50)
    """
    result = image.copy()
    h, w = result.shape[:2]

    # Validate coordinates
    if not (0 <= x < w and 0 <= y < h):
        return result

    # Ensure BGR for cv2
    if result.ndim == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    has_alpha = result.shape[2] == 4
    if has_alpha:
        bgr = result[:, :, :3]
        alpha = result[:, :, 3]
    else:
        bgr = result
        alpha = None

    # Create mask (must be 2 pixels larger)
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    # Convert color to BGR
    if len(color) == 4:
        fill_bgr = (color[2], color[1], color[0])
        fill_alpha = color[3]
    else:
        fill_bgr = (color[2], color[1], color[0])
        fill_alpha = 255

    if contiguous:
        # Standard flood fill
        lo_diff = (tolerance, tolerance, tolerance)
        hi_diff = (tolerance, tolerance, tolerance)
        flags = 4 | (255 << 8)

        cv2.floodFill(bgr, mask, (x, y), fill_bgr, lo_diff, hi_diff, flags)
    else:
        # Select all similar colors
        target_color = bgr[y, x].astype(np.int32)
        diff = np.abs(bgr.astype(np.int32) - target_color)
        similar = np.all(diff <= tolerance, axis=2)

        bgr[similar] = fill_bgr

        if alpha is not None:
            alpha[similar] = fill_alpha

    # Recombine with alpha
    if has_alpha:
        result[:, :, :3] = bgr
        if alpha is not None and not contiguous:
            result[:, :, 3] = alpha
    else:
        result = bgr

    return result
