# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - RGB to Grayscale conversion."""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

import numpy as np

# Type for array-like inputs
ArrayLike = np.ndarray | list | tuple


def rgb_to_gray(rgb: np.ndarray, method: str = "luminance") -> np.ndarray:
    """
    Convert RGB to grayscale.

    Args:
        rgb: RGB array
        method: Conversion method
            - 'luminance': Perceptual luminance (ITU-R BT.601)
            - 'lightness': Average of min and max
            - 'average': Simple average
            - 'luminosity': ITU-R BT.709

    Returns:
        Grayscale array
    """
    rgb = np.asarray(rgb, dtype=np.float64)

    has_alpha = rgb.shape[-1] == 4
    alpha: np.ndarray | None = None

    if has_alpha:
        alpha = rgb[..., 3:4]
        rgb = rgb[..., :3]

    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    if method == "luminance":
        # ITU-R BT.601 (standard NTSC)
        gray = 0.299 * r + 0.587 * g + 0.114 * b
    elif method == "lightness":
        gray = (
            np.maximum(np.maximum(r, g), b) + np.minimum(np.minimum(r, g), b)
        ) / 2
    elif method == "average":
        gray = (r + g + b) / 3
    elif method == "luminosity":
        # ITU-R BT.709 (HDTV)
        gray = 0.2126 * r + 0.7152 * g + 0.0722 * b
    else:
        raise ValueError(f"Unknown method: {method}")

    gray = gray[..., np.newaxis]

    if has_alpha and alpha is not None:
        gray = np.concatenate([gray, alpha], axis=-1)

    return gray
