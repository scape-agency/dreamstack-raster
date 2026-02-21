"""
Match Template Function
=======================

Find best match of template in image.

"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from .match_method import MatchMethod
from .match_result import MatchResult
from .utils import _get_cv2_method, _is_minimum_method


def match_template(
    image: NDArray[np.uint8],
    template: NDArray[np.uint8],
    *,
    method: MatchMethod | str = MatchMethod.CCOEFF_NORMED,
    mask: NDArray[np.uint8] | None = None,
) -> MatchResult:
    """Find best match of template in image.

    Searches for a template image within a larger image
    and returns the location of the best match.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Source image to search in.
    template : NDArray[np.uint8]
        Template to find.
    method : MatchMethod or str, optional
        Matching method. Default is CCOEFF_NORMED.
    mask : NDArray[np.uint8], optional
        Optional mask for template.

    Returns
    -------
    MatchResult
        Information about the best match.

    Examples
    --------
    >>> # Find a button in a screenshot
    >>> result = match_template(screenshot, button_template)
    >>> print(f"Found at: {result.location}")
    >>> print(f"Confidence: {result.score:.2f}")
    """
    cv_method = _get_cv2_method(method)

    # Perform template matching
    if mask is not None:
        match_map = cv2.matchTemplate(
            image,
            template,
            cv_method,
            mask=mask,
        )
    else:
        match_map = cv2.matchTemplate(
            image,
            template,
            cv_method,
        )

    # Find best match
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match_map)

    # Determine best location based on method
    if _is_minimum_method(method):
        location = min_loc
        score = 1.0 - min_val if "normed" in str(method).lower() else -min_val
    else:
        location = max_loc
        score = max_val

    # Calculate bounding box and center
    h, w = template.shape[:2]
    loc: tuple[int, int] = (int(location[0]), int(location[1]))
    bbox = (loc[0], loc[1], w, h)
    center = (loc[0] + w // 2, loc[1] + h // 2)

    return MatchResult(
        location=loc,
        score=score,
        bounding_box=bbox,
        center=center,
    )
