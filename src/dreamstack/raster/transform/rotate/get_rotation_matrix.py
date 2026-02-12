"""Get rotation matrix."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def get_rotation_matrix(
    center: tuple[float, float],
    angle: float,
    scale: float = 1.0,
) -> NDArray[np.float32]:
    """Get 2x3 rotation matrix.

    Parameters
    ----------
    center : tuple[float, float]
        Rotation center (x, y).
    angle : float
        Rotation angle in degrees.
    scale : float, optional
        Scaling factor. Default is 1.0.

    Returns
    -------
    NDArray[np.float32]
        2x3 rotation matrix.

    Examples
    --------
    >>> h, w = img.shape[:2]
    >>> matrix = get_rotation_matrix((w/2, h/2), 45, scale=0.8)
    >>> rotated = cv2.warpAffine(img, matrix, (w, h))
    """
    return cv2.getRotationMatrix2D(center, angle, scale)
