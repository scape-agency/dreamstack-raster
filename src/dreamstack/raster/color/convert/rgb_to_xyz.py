# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - RGB to XYZ conversion."""

# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

import numpy as np

from dreamstack.raster.color.spaces.color_space import ColorSpace
from dreamstack.raster.color.spaces.color_space_instances import D50, D65, sRGB

# Type for array-like inputs
ArrayLike = np.ndarray | list | tuple

BRADFORD_MATRIX = np.array(
    [
        [0.8951, 0.2664, -0.1614],
        [-0.7502, 1.7135, 0.0367],
        [0.0389, -0.0685, 1.0296],
    ]
)
BRADFORD_MATRIX_INV = np.linalg.inv(BRADFORD_MATRIX)


def _xy_to_xyz(xy: np.ndarray) -> np.ndarray:
    x, y = np.asarray(xy, dtype=np.float64)
    return np.array([x / y, 1.0, (1.0 - x - y) / y], dtype=np.float64)


def _illuminant_white_point(illuminant: str) -> np.ndarray:
    if illuminant == "D65":
        return np.asarray(D65, dtype=np.float64)
    if illuminant == "D50":
        return np.asarray(D50, dtype=np.float64)
    raise ValueError(f"Unknown illuminant: {illuminant}")


def _adaptation_matrix(
    source_white_point: np.ndarray, target_white_point: np.ndarray
) -> np.ndarray:
    source_white_point = np.asarray(source_white_point, dtype=np.float64)
    target_white_point = np.asarray(target_white_point, dtype=np.float64)

    if np.allclose(source_white_point, target_white_point):
        return np.eye(3, dtype=np.float64)

    source_xyz = _xy_to_xyz(source_white_point)
    target_xyz = _xy_to_xyz(target_white_point)

    source_lms = BRADFORD_MATRIX @ source_xyz
    target_lms = BRADFORD_MATRIX @ target_xyz
    scale = np.diag(target_lms / source_lms)

    return BRADFORD_MATRIX_INV @ scale @ BRADFORD_MATRIX


def _working_space_or_default(
    working_space: ColorSpace | None,
) -> ColorSpace:
    return sRGB if working_space is None else working_space


def rgb_to_xyz(
    rgb: np.ndarray,
    illuminant: str = "D65",
    *,
    working_space: ColorSpace | None = None,
) -> np.ndarray:
    """
    Convert RGB to CIE XYZ color space.

    Uses the supplied RGB working space, defaulting to sRGB.

    Args:
        rgb: RGB array with values in [0, 1] range
        illuminant: Reference illuminant (D65, D50)
        working_space: RGB working space describing primaries, white point,
            and transfer function

    Returns:
        XYZ array
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    working_space = _working_space_or_default(working_space)

    input_shape = rgb.shape
    has_alpha = input_shape[-1] == 4
    alpha: np.ndarray | None = None

    if has_alpha:
        alpha = rgb[..., 3:4]
        rgb = rgb[..., :3]

    linear = working_space.linearize(rgb)
    xyz = np.einsum("...j,ij->...i", linear, working_space.rgb_to_xyz_matrix)
    xyz = np.einsum(
        "...j,ij->...i",
        xyz,
        _adaptation_matrix(
            working_space.white_point,
            _illuminant_white_point(illuminant),
        ),
    )

    if has_alpha and alpha is not None:
        xyz = np.concatenate([xyz, alpha], axis=-1)

    return xyz
