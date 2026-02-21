# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Luminosity Histogram Function
=============================

Compute luminosity histogram.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .histogram import histogram


def histogram_luminosity(
    image: NDArray[np.uint8],
    bins: int = 256,
    mask: NDArray[np.uint8] | None = None,
) -> NDArray[np.float64]:
    """Compute luminosity histogram.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input image.
    bins : int, optional
        Number of bins. Default is 256.
    mask : NDArray[np.uint8], optional
        Optional mask.

    Returns
    -------
    NDArray[np.float64]
        Luminosity histogram.
    """
    if image.ndim == 3:
        # Convert to luminosity
        b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        lum = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)
    else:
        lum = image

    return histogram(lum, bins, mask)
