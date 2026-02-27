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
        Apply random offset to grid lines. Default False.
        Note: Offset/jitter is applied at placement time instead.
    max_offset : int
        Maximum random offset in pixels (equals margin). Default 50.
    empty_alpha_threshold : int
        Alpha value at or below which pixel is considered empty (padded).
        Only detects pixels we added as transparent padding. Default 0.
    generate_inbetweens : bool
        Generate horizontal and vertical in-between segments at half-positions.
        Creates ~2x more segments. Default False (superseded by fluid_grid).
    generate_diagonal_inbetweens : bool
        Generate diagonal in-between segments at cell intersections.
        Creates additional segments between the h/v in-betweens. Default False.
    fluid_grid : bool
        Use organic irregular grid instead of uniform tiles. Default True.
    size_variation : float
        Segment size variation factor (0.0-1.0). 0.3 means ±30%. Default 0.3.
    layer_count : int
        Number of overlapping segmentation passes. Default 2.
    layer_selection_ratio : float
        Fraction of segments to use from each layer (0.0-1.0). Default 0.7.
    rotation_range : float
        Maximum rotation in degrees (±). Applied at placement. Default 5.0.
    """

    segment_size: tuple[int, int] = (400, 300)
    randomize_offset: bool = False
    max_offset: int = 50
    empty_alpha_threshold: int = 0
    generate_inbetweens: bool = False
    generate_diagonal_inbetweens: bool = False
    fluid_grid: bool = True
    size_variation: float = 0.3
    layer_count: int = 2
    layer_selection_ratio: float = 0.7
    rotation_range: float = 5.0
