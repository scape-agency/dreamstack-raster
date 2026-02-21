# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Extracted Object
================

Data class representing an extracted object from an image.
"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from dreamstack.raster.analysis.contour.info import ContourInfo


@dataclass
class ExtractedObject:
    """Represents an extracted object from an image.

    Attributes
    ----------
    image : NDArray[np.uint8]
        The extracted object image (BGR or BGRA).
    original_region : tuple[int, int, int, int]
        Region from original image (x, y, width, height).
    contour : ContourInfo | None
        Contour information used for extraction.
    index : int
        Sequential index of this object.
    source_path : Path | None
        Path to source image (if loaded from file).

    Examples
    --------
    >>> obj = ExtractedObject(image, (100, 100, 200, 200), contour, 0)
    >>> print(f"Object {obj.index}: {obj.dimensions}")
    """

    image: NDArray[np.uint8]
    original_region: tuple[int, int, int, int]
    contour: ContourInfo | None = None
    index: int = 0
    source_path: Path | None = None

    @property
    def dimensions(self) -> tuple[int, int]:
        """Get image dimensions as (height, width)."""
        return self.image.shape[:2]

    @property
    def area(self) -> float:
        """Get contour area, or 0 if no contour."""
        return self.contour.area if self.contour else 0.0

    @property
    def center(self) -> tuple[float, float]:
        """Get center point in original image coordinates."""
        x, y, w, h = self.original_region
        return (x + w / 2, y + h / 2)
