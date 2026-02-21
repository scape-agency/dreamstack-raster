# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Channels last operation."""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def channels_last(image: NDArray) -> NDArray:
    """Convert from channels-first to channels-last format.

    Converts (C, H, W) to (H, W, C) for TensorFlow.

    Parameters
    ----------
    image : NDArray
        Channels-first image.

    Returns
    -------
    NDArray
        Channels-last image.
    """
    if image.ndim == 3:
        return np.transpose(image, (1, 2, 0))
    elif image.ndim == 4:
        return np.transpose(image, (0, 2, 3, 1))
    return image
