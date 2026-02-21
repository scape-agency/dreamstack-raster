# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Refine Mask
===========

Mask refinement with morphological operations and blurring.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from dreamstack.raster.effects.background.mask_refinement_config import (
    MaskRefinementConfig,
)

# =============================================================================
# Type Checking Imports
# =============================================================================

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel
    from numpy.typing import NDArray


def refine_mask(
    mask: NDArray[np.uint8],
    config: MaskRefinementConfig | None = None,
    *,
    dilate_iterations: int | None = None,
    erode_iterations: int | None = None,
    blur_size: int | None = None,
    feather_amount: int | None = None,
) -> NDArray[np.uint8]:
    """Refine a mask with morphological operations and blurring.

    Apply dilation, erosion, blurring, and feathering to improve mask edges.

    Args:
        mask: Input grayscale mask (single channel).
        config: Optional MaskRefinementConfig for settings.
        dilate_iterations: Number of dilation iterations (expands mask).
        erode_iterations: Number of erosion iterations (shrinks mask).
        blur_size: Gaussian blur kernel size (must be odd, 0 for no blur).
        feather_amount: Edge feathering in pixels.

    Returns:
        Refined grayscale mask.

    Example:
        >>> refined = refine_mask(mask, dilate_iterations=2, blur_size=5)
    """
    import cv2  # pylint: disable=import-outside-toplevel

    # Use config or defaults
    cfg = config or MaskRefinementConfig()

    # Override with keyword arguments
    d_iter = (
        dilate_iterations
        if dilate_iterations is not None
        else cfg.dilate_iterations
    )
    e_iter = (
        erode_iterations
        if erode_iterations is not None
        else cfg.erode_iterations
    )
    b_size = blur_size if blur_size is not None else cfg.blur_size
    f_amount = (
        feather_amount if feather_amount is not None else cfg.feather_amount
    )

    result = mask.copy()

    # Create kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # Apply dilation (expands white regions)
    if d_iter > 0:
        result = cv2.dilate(result, kernel, iterations=d_iter)

    # Apply erosion (shrinks white regions)
    if e_iter > 0:
        result = cv2.erode(result, kernel, iterations=e_iter)

    # Apply Gaussian blur for smoothing
    if b_size > 0:
        # Ensure kernel size is odd
        if b_size % 2 == 0:
            b_size += 1
        result = cv2.GaussianBlur(result, (b_size, b_size), 0)

    # Apply feathering (soft edge transition)
    if f_amount > 0:
        # Use distance transform for feathering
        dist = cv2.distanceTransform(result, cv2.DIST_L2, 5)
        dist = np.clip(dist / f_amount, 0, 1)
        result = (dist * 255).astype(np.uint8)

    return result
