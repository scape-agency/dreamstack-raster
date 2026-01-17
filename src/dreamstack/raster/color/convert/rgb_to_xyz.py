# -*- coding: utf-8 -*-

"""RGB to XYZ conversion."""

from __future__ import annotations

from typing import Union

import numpy as np

# Type for array-like inputs
ArrayLike = Union[np.ndarray, list, tuple]


def rgb_to_xyz(rgb: np.ndarray, illuminant: str = "D65") -> np.ndarray:
    """
    Convert RGB to CIE XYZ color space.

    Uses sRGB primaries by default.

    Args:
        rgb: RGB array with values in [0, 1] range
        illuminant: Reference illuminant (D65, D50)

    Returns:
        XYZ array
    """
    rgb = np.asarray(rgb, dtype=np.float64)

    input_shape = rgb.shape
    has_alpha = input_shape[-1] == 4

    if has_alpha:
        alpha = rgb[..., 3:4]
        rgb = rgb[..., :3]

    # Linearize sRGB
    linear = np.where(
        rgb <= 0.04045, rgb / 12.92, np.power((rgb + 0.055) / 1.055, 2.4)
    )

    # sRGB to XYZ matrix (D65)
    if illuminant == "D65":
        m = np.array(
            [
                [0.4124564, 0.3575761, 0.1804375],
                [0.2126729, 0.7151522, 0.0721750],
                [0.0193339, 0.1191920, 0.9503041],
            ]
        )
    elif illuminant == "D50":
        m = np.array(
            [
                [0.4360747, 0.3850649, 0.1430804],
                [0.2225045, 0.7168786, 0.0606169],
                [0.0139322, 0.0971045, 0.7141733],
            ]
        )
    else:
        raise ValueError(f"Unknown illuminant: {illuminant}")

    xyz = np.einsum("...j,ij->...i", linear, m)

    if has_alpha:
        xyz = np.concatenate([xyz, alpha], axis=-1)

    return xyz
