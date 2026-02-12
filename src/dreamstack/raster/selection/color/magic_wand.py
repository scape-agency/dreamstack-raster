"""
Magic Wand Selection
====================

Select contiguous regions of similar color.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from dreamstack.raster.selection.shapes.selection import Selection


def magic_wand(
    image: NDArray[np.uint8],
    x: int,
    y: int,
    *,
    tolerance: int = 32,
    contiguous: bool = True,
    anti_alias: bool = True,
    sample_all_layers: bool = False,
) -> Selection:
    """Select pixels similar to the clicked point.

    Uses flood fill algorithm for contiguous selection,
    or color distance for non-contiguous selection.

    Args:
        image: Input image (BGR or BGRA).
        x: X-coordinate of seed point.
        y: Y-coordinate of seed point.
        tolerance: Color tolerance (0-255).
        contiguous: If True, select only connected pixels.
        anti_alias: Enable anti-aliased edges.
        sample_all_layers: Ignored (for API compatibility).

    Returns:
        Selection based on color similarity.

    Example:
        >>> sel = magic_wand(image, 100, 100, tolerance=48)
        >>> masked = sel.apply_to_image(image)
    """
    h, w = image.shape[:2]

    # Get source image (BGR only for flood fill)
    if image.ndim == 2:
        src = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        src = image[:, :, :3].copy()
    else:
        src = image.copy()

    # Get seed color
    seed_color = src[y, x].astype(np.int16)

    if contiguous:
        # Use flood fill for contiguous selection
        mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

        flags = 4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY
        cv2.floodFill(
            src,
            mask,
            (x, y),
            (255, 255, 255),
            (tolerance, tolerance, tolerance),
            (tolerance, tolerance, tolerance),
            flags,
        )

        # Extract the mask (remove padding)
        result_mask = mask[1:-1, 1:-1]

    else:
        # Non-contiguous: select all similar colors
        diff = np.abs(src.astype(np.int16) - seed_color)
        color_dist = np.max(diff, axis=2)

        result_mask = np.where(color_dist <= tolerance, 255, 0).astype(np.uint8)

    # Anti-alias edges
    if anti_alias:
        # Slight blur to smooth edges
        result_mask = cv2.GaussianBlur(result_mask, (3, 3), 0.5)

    return Selection(mask=result_mask)
