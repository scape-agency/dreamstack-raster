"""
Expand Selection
================

Expand/dilate selection borders.

"""

from __future__ import annotations

import cv2

from dreamstack.raster.selection.shapes.selection import Selection


def expand(
    selection: Selection,
    pixels: int,
) -> Selection:
    """Expand the selection by a number of pixels.

    Grows the selection outward in all directions.

    Args:
        selection: Input selection.
        pixels: Number of pixels to expand by.

    Returns:
        Expanded selection.

    Example:
        >>> expanded = expand(selection, 5)  # Grow 5 pixels
    """
    if pixels <= 0:
        return selection.copy()

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (pixels * 2 + 1, pixels * 2 + 1),
    )

    expanded_mask = cv2.dilate(selection.mask, kernel)

    return Selection(mask=expanded_mask)
