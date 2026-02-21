# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Find Pattern Function
=====================

Find a pattern at multiple scales.

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
from .match_template import match_template


def find_pattern(
    image: NDArray[np.uint8],
    pattern: NDArray[np.uint8],
    scales: list[float] | None = None,
    *,
    method: MatchMethod | str = MatchMethod.CCOEFF_NORMED,
    threshold: float = 0.8,
) -> MatchResult | None:
    """Find a pattern at multiple scales.

    Searches for a template at different sizes to handle
    scale variations in the target image.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Source image.
    pattern : NDArray[np.uint8]
        Pattern to find.
    scales : list[float], optional
        Scale factors to try. Default is [0.5, 0.75, 1.0, 1.25, 1.5].
    method : MatchMethod or str, optional
        Matching method. Default is CCOEFF_NORMED.
    threshold : float, optional
        Minimum match quality. Default is 0.8.

    Returns
    -------
    MatchResult or None
        Best match across all scales, or None if no match above threshold.
    """
    if scales is None:
        scales = [0.5, 0.75, 1.0, 1.25, 1.5]

    best_match = None
    best_score = -float("inf")

    for scale in scales:
        h, w = pattern.shape[:2]
        new_shape = (int(w * scale), int(h * scale))

        if new_shape[0] < 1 or new_shape[1] < 1:
            continue
        if new_shape[0] > image.shape[1] or new_shape[1] > image.shape[0]:
            continue

        scaled_pattern = cv2.resize(pattern, new_shape)
        result = match_template(image, scaled_pattern, method=method)  # type: ignore[arg-type]

        if result.score > best_score and result.score >= threshold:
            best_score = result.score
            best_match = result

    return best_match
