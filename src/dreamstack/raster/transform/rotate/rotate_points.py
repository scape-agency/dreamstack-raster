"""Rotate points operation."""

# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def rotate_points(
    points: NDArray[np.float32],
    angle: float,
    center: tuple[float, float] = (0, 0),
) -> NDArray[np.float32]:
    """Rotate multiple points around a center.

    Parameters
    ----------
    points : NDArray[np.float32]
        Points array of shape (N, 2).
    angle : float
        Rotation angle in degrees.
    center : tuple[float, float], optional
        Center of rotation.

    Returns
    -------
    NDArray[np.float32]
        Rotated points array.
    """
    rad = np.deg2rad(angle)
    cos_val = np.cos(rad)
    sin_val = np.sin(rad)

    # Translate to origin
    translated = points - np.array(center)

    # Rotation matrix
    rotation = np.array([[cos_val, -sin_val], [sin_val, cos_val]])

    # Rotate and translate back
    rotated = translated @ rotation.T
    return rotated + np.array(center)
