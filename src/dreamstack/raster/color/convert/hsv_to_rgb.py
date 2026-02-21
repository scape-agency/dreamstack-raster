# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""HSV to RGB conversion."""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

import numpy as np

# Type for array-like inputs
ArrayLike = np.ndarray | list | tuple


def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    """
    Convert HSV to RGB color space.

    Args:
        hsv: HSV array with H in [0, 360], S and V in [0, 1]
             Shape can be (3,), (H, W, 3), or (H, W, 4)

    Returns:
        RGB array with values in [0, 1] range
    """
    hsv = np.asarray(hsv, dtype=np.float64)

    input_shape = hsv.shape
    has_alpha = input_shape[-1] == 4
    alpha: np.ndarray | None = None

    if has_alpha:
        alpha = hsv[..., 3:4]
        hsv = hsv[..., :3]

    if hsv.ndim == 1:
        hsv = hsv.reshape(1, 1, 3)
        squeeze = True
    elif hsv.ndim == 2:
        hsv = hsv.reshape(hsv.shape[0], 1, 3)
        squeeze = True
    else:
        squeeze = False

    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    h = h / 60.0
    i = np.floor(h).astype(int) % 6
    f = h - np.floor(h)

    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    rgb = np.zeros(hsv.shape)

    mask = i == 0
    rgb[mask] = np.stack([v[mask], t[mask], p[mask]], axis=-1)

    mask = i == 1
    rgb[mask] = np.stack([q[mask], v[mask], p[mask]], axis=-1)

    mask = i == 2
    rgb[mask] = np.stack([p[mask], v[mask], t[mask]], axis=-1)

    mask = i == 3
    rgb[mask] = np.stack([p[mask], q[mask], v[mask]], axis=-1)

    mask = i == 4
    rgb[mask] = np.stack([t[mask], p[mask], v[mask]], axis=-1)

    mask = i == 5
    rgb[mask] = np.stack([v[mask], p[mask], q[mask]], axis=-1)

    if has_alpha and alpha is not None:
        rgb = np.concatenate([rgb, alpha], axis=-1)

    if squeeze:
        rgb = rgb.squeeze()

    return rgb
