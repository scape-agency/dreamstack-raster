# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""Utilities for working with local color models."""

# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

import numpy as np

from dreamstack.raster.color.convert import (
    cmyk_to_rgb,
    hsl_to_rgb,
    hsv_to_rgb,
    rgb_to_cmyk,
    rgb_to_hsl,
    rgb_to_hsv,
)
from dreamstack.raster.color.models import (
    CMYKColorModel,
    HSLColorModel,
    HSVColorModel,
    RGBColorModel,
)

ColorModel = RGBColorModel | HSLColorModel | HSVColorModel | CMYKColorModel


def get_color_model(
    array: np.ndarray,
    color_space: str = "rgb",
    normalized: bool = True,
) -> ColorModel:
    """
    Create a local color model from a numpy array.

    Args:
        array: Color array of shape (3,) or (4,)
        color_space: Target color space ('rgb', 'hsl', 'hsv', 'cmyk')
        normalized: Whether input values are in normalized range

    Returns:
        Appropriate local color model
    """
    array = np.asarray(array).flatten()[:4]

    color_space = color_space.lower()

    if color_space == "rgb":
        if normalized:
            r, g, b = (
                int(array[0] * 255),
                int(array[1] * 255),
                int(array[2] * 255),
            )
        else:
            r, g, b = int(array[0]), int(array[1]), int(array[2])
        a = float(array[3]) if len(array) > 3 else 1.0
        return RGBColorModel(r, g, b, a)

    elif color_space == "hsl":
        # HSL: H in 0-360, S and L in 0-100
        if normalized:
            h = float(array[0] * 360)
            s = float(array[1] * 100)
            l = float(array[2] * 100)
        else:
            h, s, l = float(array[0]), float(array[1]), float(array[2])
        a = float(array[3]) if len(array) > 3 else 1.0
        return HSLColorModel(h, s, l, a)

    elif color_space == "hsv":
        # HSV: H in 0-360, S and V in 0-100
        if normalized:
            h = float(array[0] * 360)
            s = float(array[1] * 100)
            v = float(array[2] * 100)
        else:
            h, s, v = float(array[0]), float(array[1]), float(array[2])
        a = float(array[3]) if len(array) > 3 else 1.0
        return HSVColorModel(h, s, v, a)

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
        return CMYKColorModel(c, m, y, k)

    else:
        raise ValueError(f"Unknown color space: {color_space}")


def convert_color_model(
    color: ColorModel,
    target: str,
) -> ColorModel:
    """
    Convert a color model to a different color space.

    Args:
        color: Source color model
        target: Target color space ('rgb', 'hsl', 'hsv', 'cmyk')

    Returns:
        Converted color model
    """
    target = target.lower()

    # First convert to RGB if not already
    if isinstance(color, RGBColorModel):
        rgb = color
    elif isinstance(color, HSLColorModel):
        rgb_array = hsl_to_rgb(
            np.array(
                [color.h, color.s / 100.0, color.l / 100.0], dtype=np.float64
            )
        )
        rgb = RGBColorModel(
            int(round(rgb_array[0] * 255)),
            int(round(rgb_array[1] * 255)),
            int(round(rgb_array[2] * 255)),
            color.a,
        )
    elif isinstance(color, HSVColorModel):
        rgb_array = hsv_to_rgb(
            np.array(
                [color.h, color.s / 100.0, color.v / 100.0], dtype=np.float64
            )
        )
        rgb = RGBColorModel(
            int(round(rgb_array[0] * 255)),
            int(round(rgb_array[1] * 255)),
            int(round(rgb_array[2] * 255)),
            color.a,
        )
    elif isinstance(color, CMYKColorModel):
        rgb_array = cmyk_to_rgb(
            np.array(
                [
                    color.c / 100.0,
                    color.m / 100.0,
                    color.y / 100.0,
                    color.k / 100.0,
                ],
                dtype=np.float64,
            )
        )
        rgb = RGBColorModel(
            int(round(rgb_array[0] * 255)),
            int(round(rgb_array[1] * 255)),
            int(round(rgb_array[2] * 255)),
            color.a,
        )
    else:
        raise TypeError(f"Unknown color type: {type(color)}")

    # Now convert to target
    if target == "rgb":
        return rgb
    elif target == "hsl":
        hue, saturation, lightness = rgb_to_hsl(rgb.to_array(normalized=True))
        return HSLColorModel(hue, saturation * 100.0, lightness * 100.0, rgb.a)
    elif target == "hsv":
        hue, saturation, value = rgb_to_hsv(rgb.to_array(normalized=True))
        return HSVColorModel(hue, saturation * 100.0, value * 100.0, rgb.a)
    elif target == "cmyk":
        cyan, magenta, yellow, key = rgb_to_cmyk(rgb.to_array(normalized=True))
        return CMYKColorModel(
            cyan * 100.0,
            magenta * 100.0,
            yellow * 100.0,
            key * 100.0,
            rgb.a,
        )
    else:
        raise ValueError(f"Unknown target color space: {target}")


__all__: list[str] = [
    "get_color_model",
    "convert_color_model",
]
