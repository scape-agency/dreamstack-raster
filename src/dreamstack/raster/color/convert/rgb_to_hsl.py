# -*- coding: utf-8 -*-

"""RGB to HSL conversion."""

from __future__ import annotations

from typing import Union

import numpy as np

# Type for array-like inputs
ArrayLike = Union[np.ndarray, list, tuple]


def rgb_to_hsl(rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB to HSL color space.

    Args:
        rgb: RGB array with values in [0, 1] range

    Returns:
        HSL array with H in [0, 360], S and L in [0, 1]
    """
    rgb = np.asarray(rgb, dtype=np.float64)

    input_shape = rgb.shape
    has_alpha = input_shape[-1] == 4

    if has_alpha:
        alpha = rgb[..., 3:4]
        rgb = rgb[..., :3]

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

    # Lightness
    l = (max_rgb + min_rgb) / 2

    # Saturation
    s = np.where(
        diff == 0,
        0,
        np.where(
            l <= 0.5,
            diff / (max_rgb + min_rgb),
            diff / (2 - max_rgb - min_rgb),
        ),
    )

    # Hue (same as HSV)
    h = np.zeros_like(max_rgb)

    mask = (max_rgb == r) & (diff != 0)
    h[mask] = 60 * (((g[mask] - b[mask]) / diff[mask]) % 6)

    mask = (max_rgb == g) & (diff != 0)
    h[mask] = 60 * (((b[mask] - r[mask]) / diff[mask]) + 2)

    mask = (max_rgb == b) & (diff != 0)
    h[mask] = 60 * (((r[mask] - g[mask]) / diff[mask]) + 4)

    hsl = np.stack([h, s, l], axis=-1)

    if has_alpha:
        hsl = np.concatenate([hsl, alpha], axis=-1)

    if squeeze:
        hsl = hsl.squeeze()

    return hsl
