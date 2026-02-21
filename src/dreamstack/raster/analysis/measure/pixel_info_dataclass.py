# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - PixelInfo Dataclass
===================

Information about a pixel.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PixelInfo:
    """Information about a pixel.

    Attributes
    ----------
    x : int
        X coordinate.
    y : int
        Y coordinate.
    rgb : tuple
        RGB values.
    hsv : tuple
        HSV values.
    lab : tuple
        LAB values.
    """

    x: int
    y: int
    rgb: tuple[int, int, int]
    hsv: tuple[int, int, int]
    lab: tuple[int, int, int]
