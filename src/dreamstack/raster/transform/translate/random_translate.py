"""Random translation operation."""

# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .translate import translate


def random_translate(
    image: NDArray[np.uint8],
    max_tx: int = 50,
    max_ty: int = 50,
    *,
    border_mode: str = "constant",
    border_value: int | tuple[int, int, int] = 0,
    seed: int | None = None,
) -> NDArray[np.uint8]:
    """Apply random translation for data augmentation.

    Randomly shifts the image within specified bounds.
    Essential for training robust ML models.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input image.
    max_tx : int, optional
        Maximum translation in x direction. Default is 50.
    max_ty : int, optional
        Maximum translation in y direction. Default is 50.
    border_mode : str, optional
        Border handling mode. Default is "constant".
    border_value : int or tuple, optional
        Fill value for constant border. Default is 0.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    NDArray[np.uint8]
        Randomly translated image.

    Examples
    --------
    >>> # Random shift up to 50 pixels in any direction
    >>> augmented = random_translate(img, max_tx=50, max_ty=50)
    """
    if seed is not None:
        np.random.seed(seed)

    tx = np.random.randint(-max_tx, max_tx + 1)
    ty = np.random.randint(-max_ty, max_ty + 1)

    return translate(image, tx, ty, border_mode=border_mode, border_value=border_value)
