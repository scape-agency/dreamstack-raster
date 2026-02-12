"""
MatchResult Dataclass
=====================

Result from template matching.

"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MatchResult:
    """Result from template matching.

    Attributes
    ----------
    location : tuple[int, int]
        Top-left corner of best match (x, y).
    score : float
        Match quality score.
    bounding_box : tuple[int, int, int, int]
        Bounding box as (x, y, width, height).
    center : tuple[int, int]
        Center point of the match.
    """

    location: tuple[int, int]
    score: float
    bounding_box: tuple[int, int, int, int]
    center: tuple[int, int]
