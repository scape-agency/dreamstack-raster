"""
Similar Selection
=================

Select all pixels similar to current selection.

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

from dreamstack.raster.selection.shapes.selection import Selection


def similar(
    selection: Selection,
    image: NDArray[np.uint8],
    *,
    tolerance: int = 32,
) -> Selection:
    """Select all pixels similar to the current selection.

    Unlike grow, this selects ALL similar pixels in the image,
    not just adjacent ones.

    Args:
        selection: Input selection (used to sample colors).
        image: Source image for color comparison.
        tolerance: Color tolerance (0-255).

    Returns:
        Selection of all similar colors.

    Example:
        >>> all_similar = similar(selection, image, tolerance=32)
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

    # Sample colors from selected area
    selected_colors = src[selection.mask > 128]

    if len(selected_colors) == 0:
        return selection.copy()

    # Calculate color statistics
    mean_color = np.mean(selected_colors, axis=0)
    std_color = np.std(selected_colors, axis=0)

    # Adaptive tolerance based on variance
    effective_tolerance = np.maximum(
        std_color + tolerance,
        tolerance,
    ).astype(np.uint8)

    lower = np.maximum(0, mean_color - effective_tolerance).astype(np.uint8)
    upper = np.minimum(255, mean_color + effective_tolerance).astype(np.uint8)

    # Select all similar colors
    similar_mask = cv2.inRange(src, lower, upper)

    return Selection(mask=similar_mask)
