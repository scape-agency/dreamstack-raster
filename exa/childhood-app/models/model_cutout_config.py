"""
Cutout Configuration
====================

Configuration dataclass for bounding box cutouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


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
    cutout_mode : Literal["bbox", "contour"]
        Cutout mode. ``"bbox"`` keeps the full bounding-box rectangle
        (current behaviour). ``"contour"`` applies the YOLO segmentation
        mask as the alpha channel so background pixels outside the
        object silhouette become transparent. Default ``"bbox"``.
    """

    max_size: int = 1200
    margin: int = 50
    segment_align: bool = True
    background_color: tuple[int, int, int, int] = (0, 0, 0, 0)
    cutout_mode: Literal["bbox", "contour"] = "bbox"
