"""
Canvas Configuration
====================

Configuration dataclass for the art installation canvas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class CanvasConfig:
    """Configuration for the art installation canvas.

    Attributes
    ----------
    size : tuple[int, int]
        Canvas size (width, height). Default 8000x3000.
    background_color : tuple[int, int, int, int]
        RGBA background color. Default transparent.
    origin : Literal["bottom-left", "top-left"]
        Coordinate origin for placement. Default bottom-left.
    """

    size: tuple[int, int] = (8000, 3000)
    background_color: tuple[int, int, int, int] = (255, 255, 255, 255)
    origin: Literal["bottom-left", "top-left"] = "bottom-left"
