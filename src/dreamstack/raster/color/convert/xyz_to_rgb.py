# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""XYZ to RGB conversion."""

# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

import numpy as np

from dreamstack.raster.color.convert.rgb_to_xyz import (
    _adaptation_matrix,
    _illuminant_white_point,
    _working_space_or_default,
)
from dreamstack.raster.color.spaces.color_space import ColorSpace

# Type for array-like inputs
ArrayLike = np.ndarray | list | tuple


def xyz_to_rgb(
    xyz: np.ndarray,
    illuminant: str = "D65",
    *,
    working_space: ColorSpace | None = None,
) -> np.ndarray:
    """
    Convert CIE XYZ to RGB color space.

    Args:
        xyz: XYZ array
        illuminant: Reference illuminant (D65, D50)
        working_space: RGB working space describing primaries, white point,
            and transfer function

    Returns:
        RGB array with values in [0, 1] range
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    working_space = _working_space_or_default(working_space)

    input_shape = xyz.shape
    has_alpha = input_shape[-1] == 4
    alpha: np.ndarray | None = None

    if has_alpha:
        alpha = xyz[..., 3:4]
        xyz = xyz[..., :3]

    xyz = np.einsum(
        "...j,ij->...i",
        xyz,
        _adaptation_matrix(
            _illuminant_white_point(illuminant),
            working_space.white_point,
        ),
    )
    linear = np.einsum("...j,ij->...i", xyz, working_space.xyz_to_rgb_matrix)
    rgb = working_space.encode(linear)

    rgb = np.clip(rgb, 0, 1)

    if has_alpha and alpha is not None:
        rgb = np.concatenate([rgb, alpha], axis=-1)

    return rgb
