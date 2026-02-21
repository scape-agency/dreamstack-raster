"""Rotate point operation."""

# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

import numpy as np


def rotate_point(
    point: tuple[float, float],
    angle: float,
    center: tuple[float, float] = (0, 0),
) -> tuple[float, float]:
    """Rotate a point around a center.

    Useful for rotating coordinates (e.g., landmarks) along with images.

    Parameters
    ----------
    point : tuple[float, float]
        Point to rotate (x, y).
    angle : float
        Rotation angle in degrees.
    center : tuple[float, float], optional
        Center of rotation. Default is origin.

    Returns
    -------
    tuple[float, float]
        Rotated point coordinates.

    Examples
    --------
    >>> # Rotate landmark along with image
    >>> rotated_img = rotate(img, 45)
    >>> rotated_point = rotate_point((100, 50), 45, center=(w/2, h/2))
    """
    rad = np.deg2rad(angle)
    cos_val = np.cos(rad)
    sin_val = np.sin(rad)

    # Translate to origin
    x = point[0] - center[0]
    y = point[1] - center[1]

    # Rotate
    new_x = x * cos_val - y * sin_val
    new_y = x * sin_val + y * cos_val

    # Translate back
    return (new_x + center[0], new_y + center[1])
