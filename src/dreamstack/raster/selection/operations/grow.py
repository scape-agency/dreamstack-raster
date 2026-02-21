"""
Grow Selection
==============

Expand selection to include similar adjacent colors.

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
    # pylint: disable=import-outside-toplevel
    from numpy.typing import NDArray

from dreamstack.raster.selection.shapes.selection import Selection


def grow(
    selection: Selection,
    image: NDArray[np.uint8],
    *,
    tolerance: int = 32,
) -> Selection:
    """Grow the selection to include adjacent similar pixels.

    Expands the selection boundary to include neighboring pixels
    with colors similar to the edge of the current selection.

    Args:
        selection: Input selection.
        image: Source image for color comparison.
        tolerance: Color tolerance (0-255).

    Returns:
        Grown selection.

    Example:
        >>> grown = grow(selection, image, tolerance=48)
    """
    if selection.is_empty:
        return selection.copy()

    h, w = selection.mask.shape

    # Get source image
    if image.ndim == 2:
        src = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        src = image[:, :, :3]
    else:
        src = image

    # Find contours of current selection
    contours, _ = cv2.findContours(
        selection.mask.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    if not contours:
        return selection.copy()

    # Sample colors along selection edge
    edge_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(edge_mask, contours, -1, 255, 2)

    edge_colors = src[edge_mask > 0]

    if len(edge_colors) == 0:
        return selection.copy()

    # Calculate color range
    mean_color = np.mean(edge_colors, axis=0)

    lower = np.maximum(0, mean_color - tolerance).astype(np.uint8)
    upper = np.minimum(255, mean_color + tolerance).astype(np.uint8)

    # Find similar colors throughout image
    similar_mask = cv2.inRange(src, lower, upper)

    # Only grow into adjacent similar areas
    # Dilate the selection slightly
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(selection.mask, kernel, iterations=2)

    # Combine with similar colors
    grown_mask = cv2.bitwise_and(dilated, similar_mask)
    grown_mask = cv2.bitwise_or(selection.mask, grown_mask)

    return Selection(mask=grown_mask)
