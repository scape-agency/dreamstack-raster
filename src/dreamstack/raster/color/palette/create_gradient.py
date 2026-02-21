# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Create gradient between two colors."""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

import numpy as np

from dreamstack.raster.color.palette.color import Color
from dreamstack.raster.color.palette.palette import Palette


def create_gradient(
    start_color: Color,
    end_color: Color,
    steps: int = 10,
    color_space: str = "rgb",
) -> Palette:
    """
    Create a gradient between two colors.

    Args:
        start_color: Starting color
        end_color: Ending color
        steps: Number of color steps
        color_space: Interpolation space ('rgb', 'hsv', 'hsl', 'lab')

    Returns:
        Palette with gradient colors
    """
    colors = []

    if color_space == "rgb":
        start = start_color.to_array(include_alpha=False)
        end = end_color.to_array(include_alpha=False)

        if not start_color.normalized:
            start = start / 255
        if not end_color.normalized:
            end = end / 255

        for i in range(steps):
            t = i / (steps - 1) if steps > 1 else 0
            rgb = start + (end - start) * t
            colors.append(
                Color(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
            )

    elif color_space == "hsv":
        h1, s1, v1 = start_color.to_hsv()
        h2, s2, v2 = end_color.to_hsv()

        # Handle hue wraparound
        if abs(h2 - h1) > 180:
            if h2 > h1:
                h1 += 360
            else:
                h2 += 360

        for i in range(steps):
            t = i / (steps - 1) if steps > 1 else 0
            h = (h1 + (h2 - h1) * t) % 360
            s = s1 + (s2 - s1) * t
            v = v1 + (v2 - v1) * t
            colors.append(Color.from_hsv(h, s, v))

    elif color_space == "hsl":
        h1, s1, l1 = start_color.to_hsl()
        h2, s2, l2 = end_color.to_hsl()

        if abs(h2 - h1) > 180:
            if h2 > h1:
                h1 += 360
            else:
                h2 += 360

        for i in range(steps):
            t = i / (steps - 1) if steps > 1 else 0
            h = (h1 + (h2 - h1) * t) % 360
            s = s1 + (s2 - s1) * t
            l = l1 + (l2 - l1) * t
            colors.append(Color.from_hsl(h, s, l))

    elif color_space == "lab":
        # pylint: disable=import-outside-toplevel
        from dreamstack.raster.color.convert import lab_to_rgb, rgb_to_lab

        start_n = np.array(start_color.to_normalized()[:3])
        end_n = np.array(end_color.to_normalized()[:3])

        lab1 = rgb_to_lab(start_n)
        lab2 = rgb_to_lab(end_n)

        for i in range(steps):
            t = i / (steps - 1) if steps > 1 else 0
            lab = lab1 + (lab2 - lab1) * t
            rgb = lab_to_rgb(lab)
            rgb = np.clip(rgb, 0, 1)
            colors.append(
                Color(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
            )

    else:
        raise ValueError(f"Unknown color space: {color_space}")

    return Palette(
        colors=colors, name=f"{start_color.to_hex()}-{end_color.to_hex()}"
    )
