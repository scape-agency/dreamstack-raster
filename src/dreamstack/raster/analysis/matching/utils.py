"""
Matching Utilities
==================

Internal utility functions for template matching.

"""

from __future__ import annotations

import cv2

from .match_method import MatchMethod


def _get_cv2_method(method: MatchMethod | str) -> int:
    """Convert method to OpenCV constant."""
    if isinstance(method, MatchMethod):
        method = method.value

    mapping = {
        "sqdiff": cv2.TM_SQDIFF,
        "sqdiff_normed": cv2.TM_SQDIFF_NORMED,
        "ccorr": cv2.TM_CCORR,
        "ccorr_normed": cv2.TM_CCORR_NORMED,
        "ccoeff": cv2.TM_CCOEFF,
        "ccoeff_normed": cv2.TM_CCOEFF_NORMED,
    }
    return mapping.get(method.lower(), cv2.TM_CCOEFF_NORMED)


def _is_minimum_method(method: MatchMethod | str) -> bool:
    """Check if method uses minimum value for best match."""
    if isinstance(method, MatchMethod):
        method = method.value
    return method.lower() in ("sqdiff", "sqdiff_normed")
