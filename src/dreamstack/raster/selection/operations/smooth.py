"""
Smooth Selection
================

Smooth jagged selection edges.

"""

from __future__ import annotations

import cv2

from dreamstack.raster.selection.shapes.selection import Selection


def smooth(
    selection: Selection,
    radius: int = 3,
) -> Selection:
    """Smooth the selection edges.

    Removes jagged edges while preserving overall shape.

    Args:
        selection: Input selection.
        radius: Smoothing radius (1-100).

    Returns:
        Smoothed selection.

    Example:
        >>> cleaned = smooth(selection, 5)
    """
    if radius <= 0:
        return selection.copy()

    # Ensure odd kernel size
    kernel_size = radius * 2 + 1

    # Apply bilateral filter for edge-preserving smoothing
    smoothed_mask = cv2.bilateralFilter(
        selection.mask,
        kernel_size,
        75,
        75,
    )

    return Selection(mask=smoothed_mask)
