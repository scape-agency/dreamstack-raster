# -*- coding: utf-8 -*-

"""Convert between color spaces."""

from __future__ import annotations

import numpy as np

from dreamstack.raster.color.spaces.color_space import ColorSpace


def convert_color_space(
    image_data: np.ndarray, from_space: ColorSpace, to_space: ColorSpace
) -> np.ndarray:
    """
    Convert image data between color spaces.

    Args:
        image_data: RGB image array (float, 0-1 range)
        from_space: Source color space
        to_space: Target color space

    Returns:
        Converted image array
    """
    # Linearize
    linear = from_space.linearize(image_data)

    # Convert to XYZ
    shape = linear.shape
    flat = linear.reshape(-1, 3)
    xyz = flat @ from_space.rgb_to_xyz_matrix.T

    # Convert from XYZ to target
    target_linear = xyz @ to_space.xyz_to_rgb_matrix.T
    target_linear = target_linear.reshape(shape)

    # Apply target gamma
    return to_space.encode(np.clip(target_linear, 0, 1))
