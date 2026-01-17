# -*- coding: utf-8 -*-

"""
Model Conversions
=================

Utilities for working with dreamstack.color models.

"""

from __future__ import annotations

from typing import Union, Type

import numpy as np

# Import dreamstack.color models and conversions
from dreamstack.color import (
    RGB, HSL, HSV, CMYK,
    rgb_to_hsl, rgb_to_hsv, rgb_to_cmyk,
    hsl_to_rgb, hsv_to_rgb, cmyk_to_rgb,
)

ColorModel = Union[RGB, HSL, HSV, CMYK]


def get_color_model(
    array: np.ndarray,
    color_space: str = "rgb",
    normalized: bool = True,
) -> ColorModel:
    """
    Create a dreamstack.color model from a numpy array.

    Args:
        array: Color array of shape (3,) or (4,)
        color_space: Target color space ('rgb', 'hsl', 'hsv', 'cmyk')
        normalized: Whether input values are in normalized range

    Returns:
        Appropriate color model (RGB, HSL, HSV, or CMYK)
    """
    array = np.asarray(array).flatten()[:4]

    color_space = color_space.lower()

    if color_space == "rgb":
        if normalized:
            r, g, b = int(array[0] * 255), int(array[1] * 255), int(array[2] * 255)
        else:
            r, g, b = int(array[0]), int(array[1]), int(array[2])
        a = float(array[3]) if len(array) > 3 else 1.0
        return RGB(r, g, b, a)

    elif color_space == "hsl":
        # HSL: H in 0-360, S and L in 0-100 (dreamstack.color convention)
        if normalized:
            h = float(array[0] * 360)
            s = float(array[1] * 100)
            l = float(array[2] * 100)
        else:
            h, s, l = float(array[0]), float(array[1]), float(array[2])
        a = float(array[3]) if len(array) > 3 else 1.0
        return HSL(h, s, l, a)

    elif color_space == "hsv":
        # HSV: H in 0-360, S and V in 0-100 (dreamstack.color convention)
        if normalized:
            h = float(array[0] * 360)
            s = float(array[1] * 100)
            v = float(array[2] * 100)
        else:
            h, s, v = float(array[0]), float(array[1]), float(array[2])
        a = float(array[3]) if len(array) > 3 else 1.0
        return HSV(h, s, v, a)

    elif color_space == "cmyk":
        # CMYK: All values in 0-100
        if normalized:
            c = float(array[0] * 100)
            m = float(array[1] * 100)
            y = float(array[2] * 100)
            k = float(array[3] if len(array) > 3 else 0) * 100
        else:
            c, m, y = float(array[0]), float(array[1]), float(array[2])
            k = float(array[3]) if len(array) > 3 else 0.0
        return CMYK(c, m, y, k)

    else:
        raise ValueError(f"Unknown color space: {color_space}")


def convert_color_model(
    color: ColorModel,
    target: str,
) -> ColorModel:
    """
    Convert a color model to a different color space using dreamstack.color.

    Args:
        color: Source color model
        target: Target color space ('rgb', 'hsl', 'hsv', 'cmyk')

    Returns:
        Converted color model
    """
    target = target.lower()

    # First convert to RGB if not already
    if isinstance(color, RGB):
        rgb = color
    elif isinstance(color, HSL):
        rgb = hsl_to_rgb(color)
    elif isinstance(color, HSV):
        rgb = hsv_to_rgb(color)
    elif isinstance(color, CMYK):
        rgb = cmyk_to_rgb(color)
    else:
        raise TypeError(f"Unknown color type: {type(color)}")

    # Now convert to target
    if target == "rgb":
        return rgb
    elif target == "hsl":
        return rgb_to_hsl(rgb)
    elif target == "hsv":
        return rgb_to_hsv(rgb)
    elif target == "cmyk":
        return rgb_to_cmyk(rgb)
    else:
        raise ValueError(f"Unknown target color space: {target}")


__all__: list[str] = [
    "get_color_model",
    "convert_color_model",
]
