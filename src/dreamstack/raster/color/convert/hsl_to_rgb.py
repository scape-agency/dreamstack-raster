# -*- coding: utf-8 -*-

"""HSL to RGB conversion."""

from __future__ import annotations

from typing import Union

import numpy as np

# Type for array-like inputs
ArrayLike = Union[np.ndarray, list, tuple]


def hsl_to_rgb(hsl: np.ndarray) -> np.ndarray:
    """
    Convert HSL to RGB color space.

    Args:
        hsl: HSL array with H in [0, 360], S and L in [0, 1]

    Returns:
        RGB array with values in [0, 1] range
    """
    hsl = np.asarray(hsl, dtype=np.float64)

    input_shape = hsl.shape
    has_alpha = input_shape[-1] == 4

    if has_alpha:
        alpha = hsl[..., 3:4]
        hsl = hsl[..., :3]

    if hsl.ndim == 1:
        hsl = hsl.reshape(1, 1, 3)
        squeeze = True
    elif hsl.ndim == 2:
        hsl = hsl.reshape(hsl.shape[0], 1, 3)
        squeeze = True
    else:
        squeeze = False

    h, s, l = hsl[..., 0], hsl[..., 1], hsl[..., 2]

    c = (1 - np.abs(2 * l - 1)) * s
    h_prime = h / 60.0
    x = c * (1 - np.abs(h_prime % 2 - 1))
    m = l - c / 2

    rgb = np.zeros(hsl.shape)

    i = np.floor(h_prime).astype(int) % 6

    mask = i == 0
    rgb[mask] = np.stack([c[mask], x[mask], np.zeros_like(c[mask])], axis=-1)

    mask = i == 1
    rgb[mask] = np.stack([x[mask], c[mask], np.zeros_like(c[mask])], axis=-1)

    mask = i == 2
    rgb[mask] = np.stack([np.zeros_like(c[mask]), c[mask], x[mask]], axis=-1)

    mask = i == 3
    rgb[mask] = np.stack([np.zeros_like(c[mask]), x[mask], c[mask]], axis=-1)

    mask = i == 4
    rgb[mask] = np.stack([x[mask], np.zeros_like(c[mask]), c[mask]], axis=-1)

    mask = i == 5
    rgb[mask] = np.stack([c[mask], np.zeros_like(c[mask]), x[mask]], axis=-1)

    rgb = rgb + m[..., np.newaxis]

    if has_alpha:
        rgb = np.concatenate([rgb, alpha], axis=-1)

    if squeeze:
        rgb = rgb.squeeze()

    return rgb
