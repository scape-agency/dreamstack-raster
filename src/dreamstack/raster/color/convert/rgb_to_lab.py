# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - RGB to LAB conversion."""

# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

import numpy as np

from dreamstack.raster.color.convert.rgb_to_xyz import rgb_to_xyz
from dreamstack.raster.color.spaces.color_space import ColorSpace

# Type for array-like inputs
ArrayLike = np.ndarray | list | tuple


def rgb_to_lab(
    rgb: np.ndarray,
    illuminant: str = "D65",
    *,
    working_space: ColorSpace | None = None,
) -> np.ndarray:
    """
    Convert RGB to CIE LAB color space.

    Args:
        rgb: RGB array with values in [0, 1] range
        illuminant: Reference illuminant
        working_space: RGB working space describing primaries, white point,
            and transfer function

    Returns:
        LAB array with L in [0, 100], a and b typically in [-128, 128]
    """
    xyz = rgb_to_xyz(rgb, illuminant=illuminant, working_space=working_space)

    input_shape = xyz.shape
    has_alpha = input_shape[-1] == 4
    alpha: np.ndarray | None = None

    if has_alpha:
        alpha = xyz[..., 3:4]
        xyz = xyz[..., :3]

    # Reference white points
    if illuminant == "D65":
        ref_white = np.array([0.95047, 1.00000, 1.08883])
    elif illuminant == "D50":
        ref_white = np.array([0.96422, 1.00000, 0.82521])
    else:
        ref_white = np.array([0.95047, 1.00000, 1.08883])

    xyz_ref = xyz / ref_white

    # Lab conversion function
    epsilon = 0.008856
    kappa = 903.3

    f = np.where(
        xyz_ref > epsilon,
        np.power(xyz_ref, 1 / 3),
        (kappa * xyz_ref + 16) / 116,
    )

    l = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])

    lab = np.stack([l, a, b], axis=-1)

    if has_alpha and alpha is not None:
        lab = np.concatenate([lab, alpha], axis=-1)

    return lab
