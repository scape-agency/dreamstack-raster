# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Grayscale to RGB conversion."""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

import numpy as np

# Type for array-like inputs
ArrayLike = np.ndarray | list | tuple


def gray_to_rgb(gray: np.ndarray) -> np.ndarray:
    """
    Convert grayscale to RGB.

    Args:
        gray: Grayscale array

    Returns:
        RGB array
    """
    gray = np.asarray(gray, dtype=np.float64)

    if gray.ndim == 2:
        return np.stack([gray, gray, gray], axis=-1)
    elif gray.ndim == 3 and gray.shape[-1] == 1:
        return np.concatenate([gray, gray, gray], axis=-1)
    elif gray.ndim == 3 and gray.shape[-1] == 2:
        # Grayscale with alpha
        g, a = gray[..., 0], gray[..., 1]
        return np.stack([g, g, g, a], axis=-1)
    else:
        return gray
