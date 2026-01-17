# -*- coding: utf-8 -*-

"""
Dreamstack Raster - Grid Settings
=================================

Grid display settings.

"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GridSettings:
    """
    Grid display settings.

    Attributes:
        enabled: Whether grid is visible
        size: Grid cell size in pixels
        subdivisions: Number of subdivisions
        color: Main grid color
        subdivision_color: Subdivision color
        snap_enabled: Whether to snap to grid
    """

    enabled: bool = False
    size: int = 50
    subdivisions: int = 1
    color: tuple = (0, 0, 0, 64)
    subdivision_color: tuple = (0, 0, 0, 32)
    snap_enabled: bool = False
