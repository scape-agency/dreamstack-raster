# -*- coding: utf-8 -*-

"""RGB to CMYK conversion."""

from __future__ import annotations

from typing import Union

import numpy as np

# Type for array-like inputs
ArrayLike = Union[np.ndarray, list, tuple]


def rgb_to_cmyk(rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB to CMYK color space.

    Note: This is a naive conversion. For accurate conversion,
    use ICC profiles.

    Args:
        rgb: RGB array with values in [0, 1] range

    Returns:
        CMYK array with values in [0, 1] range
    """
    rgb = np.asarray(rgb, dtype=np.float64)

    input_shape = rgb.shape
    has_alpha = input_shape[-1] == 4

    if has_alpha:
        alpha = rgb[..., 3:4]
        rgb = rgb[..., :3]

    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    k = 1 - np.maximum(np.maximum(r, g), b)

    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.where(k < 1, (1 - r - k) / (1 - k), 0)
        m = np.where(k < 1, (1 - g - k) / (1 - k), 0)
        y = np.where(k < 1, (1 - b - k) / (1 - k), 0)

    cmyk = np.stack([c, m, y, k], axis=-1)

    if has_alpha:
        cmyk = np.concatenate([cmyk, alpha], axis=-1)

    return cmyk
