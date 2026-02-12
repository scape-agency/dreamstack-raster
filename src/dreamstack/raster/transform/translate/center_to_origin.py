"""Center to origin operation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .translate import translate


def center_to_origin(
    image: NDArray[np.uint8],
    *,
    border_mode: str = "constant",
    border_value: int | tuple[int, int, int] = 0,
) -> NDArray[np.uint8]:
    """Move image center to origin (top-left).

    Shifts the image so that the center point is at (0, 0).
    Useful for certain geometric transformations.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input image.
    border_mode : str, optional
        Border handling mode. Default is "constant".
    border_value : int or tuple, optional
        Fill value for constant border. Default is 0.

    Returns
    -------
    NDArray[np.uint8]
        Translated image with center at origin.
    """
    h, w = image.shape[:2]
    tx = -w // 2
    ty = -h // 2
    return translate(image, tx, ty, border_mode=border_mode, border_value=border_value)
