"""
Draw Matches Function
=====================

Draw match rectangles on image.

"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from .match_result import MatchResult
from .multi_match_result import MultiMatchResult


def draw_matches(
    image: NDArray[np.uint8],
    matches: MatchResult | MultiMatchResult | list[MatchResult],
    *,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    show_score: bool = True,
) -> NDArray[np.uint8]:
    """Draw match rectangles on image.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Image to draw on (will be copied).
    matches : MatchResult, MultiMatchResult, or list
        Matches to visualize.
    color : tuple[int, int, int], optional
        Rectangle color (BGR). Default is green.
    thickness : int, optional
        Line thickness. Default is 2.
    show_score : bool, optional
        Display match scores. Default is True.

    Returns
    -------
    NDArray[np.uint8]
        Image with matches drawn.
    """
    result = image.copy()

    # Handle different input types
    if isinstance(matches, MatchResult):
        match_list = [matches]
    elif isinstance(matches, MultiMatchResult):
        match_list = matches.matches
    else:
        match_list = matches

    for match in match_list:
        x, y, w, h = match.bounding_box
        cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)

        if show_score:
            text = f"{match.score:.2f}"
            cv2.putText(
                result,
                text,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

    return result
