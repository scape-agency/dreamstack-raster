# -*- coding: utf-8 -*-

"""
Dreamstack Raster - Edge Detect
===============================

Generic edge detection implementation.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
    from dreamstack.raster.filters.edge.canny import canny
    from dreamstack.raster.filters.edge.laplacian import laplacian
    from dreamstack.raster.filters.edge.prewitt import prewitt
    from dreamstack.raster.filters.edge.scharr import scharr
    from dreamstack.raster.filters.edge.sobel import sobel

    if method == "sobel":
        return sobel(image)
    elif method == "canny":
        return canny(image, threshold1, threshold2)
    elif method == "laplacian":
        return laplacian(image)
    elif method == "prewitt":
        return prewitt(image)
    elif method == "scharr":
        return scharr(image)
    else:
        raise ValueError(f"Unknown edge detection method: {method}")
