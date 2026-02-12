"""Arbitrary rotate operation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .rotate import rotate


def arbitrary_rotate(
    image: NDArray[np.uint8],
    angle: float,
    *,
    keep_aspect: bool = True,
    border_value: int | tuple[int, int, int] = 0,
) -> NDArray[np.uint8]:
    """Rotate image by arbitrary angle, preserving full content.

    Always expands canvas to contain the entire rotated image.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input image.
    angle : float
        Rotation angle in degrees.
    keep_aspect : bool, optional
        If True, maintain aspect ratio. Default is True.
    border_value : int or tuple, optional
        Background fill color. Default is 0.

    Returns
    -------
    NDArray[np.uint8]
        Rotated image with expanded canvas.
    """
    return rotate(image, angle, expand=True, border_value=border_value)
