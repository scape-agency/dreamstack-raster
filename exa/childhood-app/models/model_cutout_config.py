"""
Cutout Configuration
====================

Configuration dataclass for bounding box cutouts.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CutoutConfig:
    """Configuration for bounding box cutouts.

    Attributes
    ----------
    max_size : int
        Maximum size for largest dimension. Default 1200.
    margin : int
        Margin around bounding box in pixels. Default 50.
    segment_align : bool
        Align smallest dimension to segment size multiple. Default True.
    background_color : tuple[int, int, int, int]
        RGBA background color for padding. Default transparent.
    """

    max_size: int = 1200
    margin: int = 50
    segment_align: bool = True
    background_color: tuple[int, int, int, int] = (0, 0, 0, 0)
