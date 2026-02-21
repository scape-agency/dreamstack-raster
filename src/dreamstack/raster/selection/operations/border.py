"""
Border Selection
================

Create a border selection from existing selection.

"""

# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

import cv2
import numpy as np

from dreamstack.raster.selection.shapes.selection import Selection


def border(
    selection: Selection,
    width: int,
) -> Selection:
    """Create a border selection around the current selection.

    Converts the selection to a border of specified width.

    Args:
        selection: Input selection.
        width: Border width in pixels.

    Returns:
        Border selection.

    Example:
        >>> # Create 5px border around shape
        >>> border_sel = border(selection, 5)
    """
    if width <= 0:
        return Selection(mask=np.zeros_like(selection.mask))

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (width * 2 + 1, width * 2 + 1),
    )

    # Dilate and erode to get border
    dilated = cv2.dilate(selection.mask, kernel)
    eroded = cv2.erode(selection.mask, kernel)

    border_mask = cv2.subtract(dilated, eroded)

    return Selection(mask=border_mask)
