# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Edge Detect
===============================

Generic edge detection implementation.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from typing import TYPE_CHECKING

# =============================================================================
# Type Checking Imports
# =============================================================================

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.image import Image


def edge_detect(
    image: Image,
    method: str = "sobel",
    threshold1: float = 100,
    threshold2: float = 200,
) -> Image:
    """
    Detect edges in image.

    Args:
        image: Input image
        method: Detection method ('sobel', 'canny', 'laplacian', 'prewitt', 'scharr')
        threshold1: Lower threshold (for Canny)
        threshold2: Upper threshold (for Canny)

    Returns:
        Edge-detected image
    """
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.filters.edge.canny import canny

    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.filters.edge.laplacian import laplacian

    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.filters.edge.prewitt import prewitt

    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.filters.edge.scharr import scharr

    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.filters.edge.sobel import sobel

    if method == "sobel":
        return sobel(image)
    if method == "canny":
        return canny(image, threshold1, threshold2)
    if method == "laplacian":
        return laplacian(image)
    if method == "prewitt":
        return prewitt(image)
    if method == "scharr":
        return scharr(image)
    raise ValueError(f"Unknown edge detection method: {method}")
