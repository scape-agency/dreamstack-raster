# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Channels first operation."""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def channels_first(image: NDArray) -> NDArray:
    """
    Convert from channels-last to channels-first format.

    Converts (H, W, C) to (C, H, W) for PyTorch.

    Parameters
    ----------
    image : NDArray
        Channels-last image.

    Returns
    -------
    NDArray
        Channels-first image.
    """
    if image.ndim == 3:
        return np.transpose(image, (2, 0, 1))
    elif image.ndim == 4:
        return np.transpose(image, (0, 3, 1, 2))
    return image
