"""
Segment Configuration
=====================

Configuration dataclass for grid segmentation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SegmentConfig:
    """Configuration for grid segmentation.

    Attributes
    ----------
    segment_size : tuple[int, int]
        Target segment size (width, height). Default 400x300.
    randomize_offset : bool
        Apply random offset to grid lines. Default True.
    max_offset : int
        Maximum random offset in pixels (equals margin). Default 50.
    empty_alpha_threshold : int
        Alpha value at or below which pixel is considered empty (padded).
        Only detects pixels we added as transparent padding. Default 0.
    generate_inbetweens : bool
        Generate horizontal and vertical in-between segments at half-positions.
        Creates ~2x more segments. Default False.
    """

    segment_size: tuple[int, int] = (400, 300)
    randomize_offset: bool = True
    max_offset: int = 50
    empty_alpha_threshold: int = 0
    generate_inbetweens: bool = False
