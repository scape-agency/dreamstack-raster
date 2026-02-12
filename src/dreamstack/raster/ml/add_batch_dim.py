"""Add batch dimension operation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def add_batch_dim(image: NDArray) -> NDArray:
    """Add batch dimension for single image.

    Converts (H, W, C) to (1, H, W, C).

    Parameters
    ----------
    image : NDArray
        Single image.

    Returns
    -------
    NDArray
        Image with batch dimension.
    """
    return image[np.newaxis, ...]
