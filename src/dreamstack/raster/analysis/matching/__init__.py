# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Template Matching Module
========================

Template matching for object detection and localization.
Find occurrences of a template image within a larger image.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from .create_template_mask import create_template_mask
from .draw_matches import draw_matches
from .find_pattern import find_pattern
from .highlight_match import highlight_match
from .match_method import MatchMethod
from .match_result import MatchResult
from .match_template import match_template
from .match_template_multi import match_template_multi
from .multi_match_result import MultiMatchResult

__all__ = [
    "MatchMethod",
    "MatchResult",
    "MultiMatchResult",
    "match_template",
    "match_template_multi",
    "draw_matches",
    "highlight_match",
    "create_template_mask",
    "find_pattern",
]
