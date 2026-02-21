# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - CMYK to RGB conversion."""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

import numpy as np

# Type for array-like inputs
ArrayLike = np.ndarray | list | tuple


def cmyk_to_rgb(cmyk: np.ndarray) -> np.ndarray:
    """
    Convert CMYK to RGB color space.

    Args:
        cmyk: CMYK array with values in [0, 1] range

    Returns:
        RGB array with values in [0, 1] range
    """
    cmyk = np.asarray(cmyk, dtype=np.float64)

    input_shape = cmyk.shape
    has_alpha = input_shape[-1] == 5
    alpha: np.ndarray | None = None

    if has_alpha:
        alpha = cmyk[..., 4:5]
        cmyk = cmyk[..., :4]

    c, m, y, k = cmyk[..., 0], cmyk[..., 1], cmyk[..., 2], cmyk[..., 3]

    r = (1 - c) * (1 - k)
    g = (1 - m) * (1 - k)
    b = (1 - y) * (1 - k)

    rgb = np.stack([r, g, b], axis=-1)

    if has_alpha and alpha is not None:
        rgb = np.concatenate([rgb, alpha], axis=-1)

    return rgb
