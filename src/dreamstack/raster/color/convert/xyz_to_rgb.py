"""XYZ to RGB conversion."""

from __future__ import annotations

import numpy as np

# Type for array-like inputs
ArrayLike = np.ndarray | list | tuple


def xyz_to_rgb(xyz: np.ndarray, illuminant: str = "D65") -> np.ndarray:
    """
    Convert CIE XYZ to RGB color space.

    Args:
        xyz: XYZ array
        illuminant: Reference illuminant (D65, D50)

    Returns:
        RGB array with values in [0, 1] range
    """
    xyz = np.asarray(xyz, dtype=np.float64)

    input_shape = xyz.shape
    has_alpha = input_shape[-1] == 4
    alpha: np.ndarray | None = None

    if has_alpha:
        alpha = xyz[..., 3:4]
        xyz = xyz[..., :3]

    # XYZ to sRGB matrix
    if illuminant == "D65":
        m = np.array(
            [
                [3.2404542, -1.5371385, -0.4985314],
                [-0.9692660, 1.8760108, 0.0415560],
                [0.0556434, -0.2040259, 1.0572252],
            ]
        )
    elif illuminant == "D50":
        m = np.array(
            [
                [3.1338561, -1.6168667, -0.4906146],
                [-0.9787684, 1.9161415, 0.0334540],
                [0.0719453, -0.2289914, 1.4052427],
            ]
        )
    else:
        raise ValueError(f"Unknown illuminant: {illuminant}")

    linear = np.einsum("...j,ij->...i", xyz, m)

    # Apply sRGB gamma
    rgb = np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.power(np.maximum(linear, 0), 1 / 2.4) - 0.055,
    )

    rgb = np.clip(rgb, 0, 1)

    if has_alpha and alpha is not None:
        rgb = np.concatenate([rgb, alpha], axis=-1)

    return rgb
