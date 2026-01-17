# -*- coding: utf-8 -*-

"""
Dreamstack Raster - Canvas Background
=====================================

Background settings for canvas.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class CanvasBackground:
    """
    Canvas background settings.

    Attributes:
        type: Background type (transparent, color, checker)
        color1: Primary color
        color2: Secondary color (for checker pattern)
        checker_size: Size of checker squares
    """

    type: str = "transparent"  # transparent, color, checker
    color1: Tuple[int, int, int, int] = (255, 255, 255, 255)
    color2: Tuple[int, int, int, int] = (204, 204, 204, 255)
    checker_size: int = 16
