"""
MultiMatchResult Dataclass
==========================

Results from multi-template matching.

"""

from __future__ import annotations

from dataclasses import dataclass, field

from .match_result import MatchResult


@dataclass
class MultiMatchResult:
    """Results from multi-template matching.

    Attributes
    ----------
    matches : list[MatchResult]
        List of all matches found.
    count : int
        Number of matches.
    """

    matches: list[MatchResult] = field(default_factory=list)
    count: int = 0
