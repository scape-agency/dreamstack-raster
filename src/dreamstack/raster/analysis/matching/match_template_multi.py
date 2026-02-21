# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Match Template Multi Function
=============================

Find all occurrences of template in image.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from .match_method import MatchMethod
from .match_result import MatchResult
from .multi_match_result import MultiMatchResult
from .utils import _get_cv2_method, _is_minimum_method


def match_template_multi(
    image: NDArray[np.uint8],
    template: NDArray[np.uint8],
    *,
    method: MatchMethod | str = MatchMethod.CCOEFF_NORMED,
    threshold: float = 0.8,
    max_matches: int | None = None,
    min_distance: int = 10,
) -> MultiMatchResult:
    """Find all occurrences of template in image.

    Searches for multiple instances of a template,
    using non-maximum suppression to avoid overlaps.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Source image.
    template : NDArray[np.uint8]
        Template to find.
    method : MatchMethod or str, optional
        Matching method. Default is CCOEFF_NORMED.
    threshold : float, optional
        Minimum match quality (0-1). Default is 0.8.
    max_matches : int, optional
        Maximum number of matches to return.
    min_distance : int, optional
        Minimum distance between match centers. Default is 10.

    Returns
    -------
    MultiMatchResult
        All matches found.

    Examples
    --------
    >>> # Find all instances of an icon
    >>> results = match_template_multi(img, icon, threshold=0.9)
    >>> print(f"Found {results.count} matches")
    """
    cv_method = _get_cv2_method(method)
    h, w = template.shape[:2]

    # Perform template matching
    match_map = cv2.matchTemplate(image, template, cv_method)

    # Find matches above threshold
    is_minimum = _is_minimum_method(method)
    if is_minimum:
        # For SQDIFF methods, lower is better
        if "normed" in str(method).lower():
            locations = np.where(match_map <= (1.0 - threshold))
            scores = 1.0 - match_map[locations]
        else:
            # Non-normalized, use negative threshold
            locations = np.where(match_map <= threshold)
            scores = -match_map[locations]
    else:
        locations = np.where(match_map >= threshold)
        scores = match_map[locations]

    # Create list of matches
    matches = []
    for pt, score in zip(zip(*locations[::-1]), scores):
        bbox = (pt[0], pt[1], w, h)
        center = (pt[0] + w // 2, pt[1] + h // 2)
        matches.append(
            MatchResult(
                location=pt,
                score=float(score),
                bounding_box=bbox,
                center=center,
            )
        )

    # Sort by score (highest first)
    matches.sort(key=lambda m: m.score, reverse=True)

    # Apply non-maximum suppression
    if min_distance > 0:
        filtered = []
        for match in matches:
            should_add = True
            for existing in filtered:
                dist = np.sqrt(
                    (match.center[0] - existing.center[0]) ** 2
                    + (match.center[1] - existing.center[1]) ** 2
                )
                if dist < min_distance:
                    should_add = False
                    break
            if should_add:
                filtered.append(match)
        matches = filtered

    # Limit number of matches
    if max_matches is not None:
        matches = matches[:max_matches]

    return MultiMatchResult(matches=matches, count=len(matches))
