"""Rotate operation."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def rotate(
    image: NDArray[np.uint8],
    angle: float,
    *,
    center: tuple[float, float] | None = None,
    scale: float = 1.0,
    border_mode: str = "constant",
    border_value: int | tuple[int, int, int] = 0,
    expand: bool = False,
) -> NDArray[np.uint8]:
    """Rotate image by arbitrary angle.

    Rotates the image around a center point by the specified angle.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input image (H, W) or (H, W, C).
    angle : float
        Rotation angle in degrees. Positive = counter-clockwise.
    center : tuple[float, float], optional
        Rotation center (x, y). Default is image center.
    scale : float, optional
        Scaling factor to apply during rotation. Default is 1.0.
    border_mode : str, optional
        How to fill empty areas:
        - "constant": Fill with border_value (default)
        - "reflect": Reflect at border
        - "replicate": Replicate edge pixels
    border_value : int or tuple, optional
        Fill value for constant border. Default is 0.
    expand : bool, optional
        If True, expand output to contain full rotated image.
        Default is False (maintain original size).

    Returns
    -------
    NDArray[np.uint8]
        Rotated image.

    Examples
    --------
    >>> # Rotate 45 degrees counter-clockwise
    >>> rotated = rotate(img, 45)

    >>> # Rotate 90 degrees clockwise around top-left
    >>> rotated = rotate(img, -90, center=(0, 0))

    >>> # Rotate and scale down by 50%
    >>> rotated = rotate(img, 180, scale=0.5)
    """
    h, w = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

    # Determine output size
    if expand:
        # Calculate new bounding box
        cos_val = abs(rotation_matrix[0, 0])
        sin_val = abs(rotation_matrix[0, 1])
        new_w = int((h * sin_val) + (w * cos_val))
        new_h = int((h * cos_val) + (w * sin_val))
        # Adjust center
        rotation_matrix[0, 2] += (new_w - w) / 2
        rotation_matrix[1, 2] += (new_h - h) / 2
        output_size = (new_w, new_h)
    else:
        output_size = (w, h)

    # Get border mode
    border_modes = {
        "constant": cv2.BORDER_CONSTANT,
        "reflect": cv2.BORDER_REFLECT,
        "replicate": cv2.BORDER_REPLICATE,
    }
    border = border_modes.get(border_mode, cv2.BORDER_CONSTANT)

    return cv2.warpAffine(
        image,
        rotation_matrix,
        output_size,
        borderMode=border,
        borderValue=border_value,
    )
