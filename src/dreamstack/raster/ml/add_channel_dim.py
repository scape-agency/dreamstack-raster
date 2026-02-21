# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Add channel dimension operation."""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def add_channel_dim(image: NDArray) -> NDArray:
    """Add channel dimension for single-channel images.

    Ensures images have shape (H, W, C) for model input.

    Parameters
    ----------
    image : NDArray
        Input image.

    Returns
    -------
    NDArray
        Image with channel dimension.
    """
    if image.ndim == 2:
        return image[:, :, np.newaxis]
    return image
