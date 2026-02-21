# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - RGB to HSV conversion."""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

import numpy as np

# Type for array-like inputs
ArrayLike = np.ndarray | list | tuple


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB to HSV color space.

    Args:
        rgb: RGB array with values in [0, 1] range
             Shape can be (3,), (H, W, 3), or (H, W, 4)

    Returns:
        HSV array with H in [0, 360], S and V in [0, 1]
    """
    rgb = np.asarray(rgb, dtype=np.float64)

    # Handle different input shapes
    input_shape = rgb.shape
    has_alpha = input_shape[-1] == 4
    alpha: np.ndarray | None = None

    if has_alpha:
        alpha = rgb[..., 3:4]
        rgb = rgb[..., :3]

    # Ensure 3D for vectorized operations
    if rgb.ndim == 1:
        rgb = rgb.reshape(1, 1, 3)
        squeeze = True
    elif rgb.ndim == 2:
        rgb = rgb.reshape(rgb.shape[0], 1, 3)
        squeeze = True
    else:
        squeeze = False

    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    max_rgb = np.maximum(np.maximum(r, g), b)
    min_rgb = np.minimum(np.minimum(r, g), b)
    diff = max_rgb - min_rgb

    # Value
    v = max_rgb

    # Saturation
    s = np.where(max_rgb != 0, diff / max_rgb, 0)

    # Hue
    h = np.zeros_like(max_rgb)

    # Where max is r
    mask = (max_rgb == r) & (diff != 0)
    h[mask] = 60 * (((g[mask] - b[mask]) / diff[mask]) % 6)

    # Where max is g
    mask = (max_rgb == g) & (diff != 0)
    h[mask] = 60 * (((b[mask] - r[mask]) / diff[mask]) + 2)

    # Where max is b
    mask = (max_rgb == b) & (diff != 0)
    h[mask] = 60 * (((r[mask] - g[mask]) / diff[mask]) + 4)

    hsv = np.stack([h, s, v], axis=-1)

    if has_alpha and alpha is not None:
        hsv = np.concatenate([hsv, alpha], axis=-1)

    if squeeze:
        hsv = hsv.squeeze()

    return hsv
