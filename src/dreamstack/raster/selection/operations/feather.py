"""
Feather Selection
=================

Apply soft edges to selection.

"""

from __future__ import annotations

import cv2

from dreamstack.raster.selection.shapes.selection import Selection


def feather(
    selection: Selection,
    radius: float,
) -> Selection:
    """Apply feathering (soft edges) to the selection.

    Creates a gradual transition at the selection edges.

    Args:
        selection: Input selection.
        radius: Feather radius in pixels.

    Returns:
        Feathered selection.

    Example:
        >>> soft = feather(selection, 10)  # 10px feather
    """
    if radius <= 0:
        return selection.copy()

    # Use Gaussian blur for smooth feathering
    feathered_mask = cv2.GaussianBlur(
        selection.mask,
        (0, 0),
        radius,
    )

    return Selection(mask=feathered_mask)
