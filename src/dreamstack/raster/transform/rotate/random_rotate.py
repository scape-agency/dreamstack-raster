"""Random rotate operation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .rotate import rotate


def random_rotate(
    image: NDArray[np.uint8],
    max_angle: float = 30.0,
    *,
    seed: int | None = None,
    border_value: int | tuple[int, int, int] = 0,
) -> NDArray[np.uint8]:
    """Apply random rotation for data augmentation.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input image.
    max_angle : float, optional
        Maximum rotation angle in degrees. Default is 30.
    seed : int, optional
        Random seed for reproducibility.
    border_value : int or tuple, optional
        Background fill. Default is 0.

    Returns
    -------
    NDArray[np.uint8]
        Randomly rotated image.

    Examples
    --------
    >>> # Random rotation between -30 and +30 degrees
    >>> augmented = random_rotate(img, max_angle=30)
    """
    if seed is not None:
        np.random.seed(seed)

    angle = np.random.uniform(-max_angle, max_angle)
    return rotate(image, angle, border_value=border_value)
