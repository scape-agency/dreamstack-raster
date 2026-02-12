"""
Contract Selection
==================

Contract/erode selection borders.

"""

from __future__ import annotations

import cv2

from dreamstack.raster.selection.shapes.selection import Selection


def contract(
    selection: Selection,
    pixels: int,
) -> Selection:
    """Contract the selection by a number of pixels.

    Shrinks the selection inward from all borders.

    Args:
        selection: Input selection.
        pixels: Number of pixels to contract by.

    Returns:
        Contracted selection.

    Example:
        >>> smaller = contract(selection, 3)  # Shrink 3 pixels
    """
    if pixels <= 0:
        return selection.copy()

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (pixels * 2 + 1, pixels * 2 + 1),
    )

    contracted_mask = cv2.erode(selection.mask, kernel)

    return Selection(mask=contracted_mask)
