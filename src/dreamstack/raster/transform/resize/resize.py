"""Core resize operation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from ._get_cv2_interpolation import _get_cv2_interpolation

if TYPE_CHECKING:
    from numpy.typing import NDArray

Interpolation = Literal["nearest", "linear", "cubic", "lanczos", "area"]


def resize(
    image: NDArray[np.uint8],
    size: tuple[int, int],
    *,
    interpolation: Interpolation = "lanczos",
) -> NDArray[np.uint8]:
    """Resize image to exact dimensions.

    Args:
        image: Input image.
        size: Target (width, height).
        interpolation: Interpolation method.

    Returns:
        Resized image.

    Example:
        >>> resized = resize(image, (800, 600))
    """
    import cv2

    interp = _get_cv2_interpolation(interpolation)
    return np.asarray(cv2.resize(image, size, interpolation=interp), dtype=np.uint8)
