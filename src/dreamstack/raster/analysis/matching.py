# -*- coding: utf-8 -*-

"""
Template Matching Operations
============================

Template matching for object detection and localization.
Find occurrences of a template image within a larger image.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from numpy.typing import NDArray


class MatchMethod(str, Enum):
    """Template matching methods.

    Attributes
    ----------
    SQDIFF : Squared difference (best match = minimum value)
    SQDIFF_NORMED : Normalized squared difference
    CCORR : Cross correlation (best match = maximum value)
    CCORR_NORMED : Normalized cross correlation
    CCOEFF : Coefficient correlation (best match = maximum value)
    CCOEFF_NORMED : Normalized coefficient correlation (recommended)
    """

    SQDIFF = "sqdiff"
    SQDIFF_NORMED = "sqdiff_normed"
    CCORR = "ccorr"
    CCORR_NORMED = "ccorr_normed"
    CCOEFF = "ccoeff"
    CCOEFF_NORMED = "ccoeff_normed"


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

    location: Tuple[int, int]
    score: float
    bounding_box: Tuple[int, int, int, int]
    center: Tuple[int, int]


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

    matches: List[MatchResult] = field(default_factory=list)
    count: int = 0


def _get_cv2_method(method: Union[MatchMethod, str]) -> int:
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


def _is_minimum_method(method: Union[MatchMethod, str]) -> bool:
    """Check if method uses minimum value for best match."""
    if isinstance(method, MatchMethod):
        method = method.value
    return method.lower() in ("sqdiff", "sqdiff_normed")


def match_template(
    image: NDArray[np.uint8],
    template: NDArray[np.uint8],
    *,
    method: Union[MatchMethod, str] = MatchMethod.CCOEFF_NORMED,
    mask: Optional[NDArray[np.uint8]] = None,
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
        match_map = cv2.matchTemplate(image, template, cv_method, mask=mask)
    else:
        match_map = cv2.matchTemplate(image, template, cv_method)

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
    bbox = (location[0], location[1], w, h)
    center = (location[0] + w // 2, location[1] + h // 2)

    return MatchResult(
        location=location,
        score=score,
        bounding_box=bbox,
        center=center,
    )


def match_template_multi(
    image: NDArray[np.uint8],
    template: NDArray[np.uint8],
    *,
    method: Union[MatchMethod, str] = MatchMethod.CCOEFF_NORMED,
    threshold: float = 0.8,
    max_matches: Optional[int] = None,
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


def draw_matches(
    image: NDArray[np.uint8],
    matches: Union[MatchResult, MultiMatchResult, List[MatchResult]],
    *,
    color: Tuple[int, int, int] = (0, 255, 0),
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


def create_template_mask(
    template: NDArray[np.uint8],
    transparent_color: Optional[Tuple[int, int, int]] = None,
    threshold: int = 10,
) -> NDArray[np.uint8]:
    """Create mask from template for transparent matching.

    Parameters
    ----------
    template : NDArray[np.uint8]
        Template image.
    transparent_color : tuple[int, int, int], optional
        Color to treat as transparent. If None, uses top-left pixel.
    threshold : int, optional
        Color matching threshold. Default is 10.

    Returns
    -------
    NDArray[np.uint8]
        Binary mask (255 = use, 0 = ignore).
    """
    if transparent_color is None:
        # Use top-left corner pixel as transparent color
        if template.ndim == 3:
            transparent_color = tuple(template[0, 0])
        else:
            transparent_color = template[0, 0]

    if template.ndim == 2:
        # Grayscale
        diff = np.abs(template.astype(np.int32) - transparent_color)
        mask = (diff > threshold).astype(np.uint8) * 255
    else:
        # Color
        diff = np.sqrt(
            np.sum(
                (template.astype(np.float32) - np.array(transparent_color))
                ** 2,
                axis=2,
            )
        )
        mask = (diff > threshold).astype(np.uint8) * 255

    return mask


def find_pattern(
    image: NDArray[np.uint8],
    pattern: NDArray[np.uint8],
    scales: Optional[List[float]] = None,
    *,
    method: Union[MatchMethod, str] = MatchMethod.CCOEFF_NORMED,
    threshold: float = 0.8,
) -> Optional[MatchResult]:
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
        result = match_template(image, scaled_pattern, method=method)

        if result.score > best_score and result.score >= threshold:
            best_score = result.score
            best_match = result

    return best_match
