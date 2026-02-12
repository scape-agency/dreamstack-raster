"""
Highlight Match Function
========================

Highlight matched region by dimming the rest.

"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .match_result import MatchResult


def highlight_match(
    image: NDArray[np.uint8],
    match: MatchResult,
    *,
    dim_factor: float = 0.5,
) -> NDArray[np.uint8]:
    """Highlight matched region by dimming the rest.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Source image (will be copied).
    match : MatchResult
        Match to highlight.
    dim_factor : float, optional
        How much to dim non-matched areas (0-1). Default is 0.5.

    Returns
    -------
    NDArray[np.uint8]
        Image with match highlighted.
    """
    result = (image.astype(np.float32) * dim_factor).astype(np.uint8)

    x, y, w, h = match.bounding_box
    result[y : y + h, x : x + w] = image[y : y + h, x : x + w]

    return result
