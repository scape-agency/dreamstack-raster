# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""Universal color space conversion."""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

import numpy as np

from dreamstack.raster.color.convert.cmyk_to_rgb import cmyk_to_rgb
from dreamstack.raster.color.convert.gray_to_rgb import gray_to_rgb
from dreamstack.raster.color.convert.hsl_to_rgb import hsl_to_rgb
from dreamstack.raster.color.convert.hsv_to_rgb import hsv_to_rgb
from dreamstack.raster.color.convert.lab_to_rgb import lab_to_rgb
from dreamstack.raster.color.convert.rgb_to_cmyk import rgb_to_cmyk
from dreamstack.raster.color.convert.rgb_to_gray import rgb_to_gray
from dreamstack.raster.color.convert.rgb_to_hsl import rgb_to_hsl
from dreamstack.raster.color.convert.rgb_to_hsv import rgb_to_hsv
from dreamstack.raster.color.convert.rgb_to_lab import rgb_to_lab
from dreamstack.raster.color.convert.rgb_to_xyz import rgb_to_xyz
from dreamstack.raster.color.convert.xyz_to_rgb import xyz_to_rgb

# Type for array-like inputs
ArrayLike = np.ndarray | list | tuple


def convert_color(
    color: np.ndarray, from_space: str, to_space: str, **kwargs
) -> np.ndarray:
    """
    Convert color between any supported color spaces.

    Args:
        color: Color array
        from_space: Source color space (rgb, hsv, hsl, lab, xyz, cmyk, gray)
        to_space: Target color space
        **kwargs: Additional arguments for conversion

    Returns:
        Converted color array
    """
    from_space = from_space.lower()
    to_space = to_space.lower()

    if from_space == to_space:
        return color

    # Convert to RGB first
    if from_space == "rgb":
        rgb = color
    elif from_space == "hsv":
        rgb = hsv_to_rgb(color)
    elif from_space == "hsl":
        rgb = hsl_to_rgb(color)
    elif from_space == "lab":
        rgb = lab_to_rgb(color, **kwargs)
    elif from_space == "xyz":
        rgb = xyz_to_rgb(color, **kwargs)
    elif from_space == "cmyk":
        rgb = cmyk_to_rgb(color)
    elif from_space in ("gray", "grey"):
        rgb = gray_to_rgb(color)
    else:
        raise ValueError(f"Unknown source color space: {from_space}")

    # Convert from RGB to target
    if to_space == "rgb":
        return rgb
    elif to_space == "hsv":
        return rgb_to_hsv(rgb)
    elif to_space == "hsl":
        return rgb_to_hsl(rgb)
    elif to_space == "lab":
        return rgb_to_lab(rgb, **kwargs)
    elif to_space == "xyz":
        return rgb_to_xyz(rgb, **kwargs)
    elif to_space == "cmyk":
        return rgb_to_cmyk(rgb)
    elif to_space in ("gray", "grey"):
        return rgb_to_gray(rgb, **kwargs)
    else:
        raise ValueError(f"Unknown target color space: {to_space}")
