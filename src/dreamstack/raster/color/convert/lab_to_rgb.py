# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - LAB to RGB conversion."""

# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

import numpy as np

from dreamstack.raster.color.convert.xyz_to_rgb import xyz_to_rgb
from dreamstack.raster.color.spaces.color_space import ColorSpace

# Type for array-like inputs
ArrayLike = np.ndarray | list | tuple


def lab_to_rgb(
    lab: np.ndarray,
    illuminant: str = "D65",
    *,
    working_space: ColorSpace | None = None,
) -> np.ndarray:
    """
    Convert CIE LAB to RGB color space.

    Args:
        lab: LAB array
        illuminant: Reference illuminant
        working_space: RGB working space describing primaries, white point,
            and transfer function

    Returns:
        RGB array with values in [0, 1] range
    """
    lab = np.asarray(lab, dtype=np.float64)

    input_shape = lab.shape
    has_alpha = input_shape[-1] == 4
    alpha: np.ndarray | None = None

    if has_alpha:
        alpha = lab[..., 3:4]
        lab = lab[..., :3]

    l, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

    fy = (l + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200

    epsilon = 0.008856
    kappa = 903.3

    xr = np.where(fx**3 > epsilon, fx**3, (116 * fx - 16) / kappa)
    yr = np.where(l > kappa * epsilon, fy**3, l / kappa)
    zr = np.where(fz**3 > epsilon, fz**3, (116 * fz - 16) / kappa)

    # Reference white points
    if illuminant == "D65":
        ref_white = np.array([0.95047, 1.00000, 1.08883])
    elif illuminant == "D50":
        ref_white = np.array([0.96422, 1.00000, 0.82521])
    else:
        ref_white = np.array([0.95047, 1.00000, 1.08883])

    xyz = np.stack([xr, yr, zr], axis=-1) * ref_white

    if has_alpha and alpha is not None:
        xyz = np.concatenate([xyz, alpha], axis=-1)

    return xyz_to_rgb(xyz, illuminant=illuminant, working_space=working_space)
