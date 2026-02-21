# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Mask Refinement Configuration
=============================

Configuration dataclass for mask refinement operations.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MaskRefinementConfig:
    """Configuration for mask refinement operations.

    Attributes:
        dilate_iterations: Number of dilation iterations.
        erode_iterations: Number of erosion iterations.
        blur_size: Gaussian blur kernel size (0 for no blur).
        feather_amount: Edge feathering amount.
        threshold: Threshold value for binary mask (None for no thresholding).
    """

    dilate_iterations: int = 0
    erode_iterations: int = 0
    blur_size: int = 0
    feather_amount: int = 0
    threshold: int | None = None
