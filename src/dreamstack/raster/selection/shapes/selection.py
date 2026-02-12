"""
Selection Class
===============

Core selection class representing a selection mask.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class Selection:
    """Represents an image selection as a mask.

    The selection is stored as a grayscale mask where:
    - 255 = fully selected
    - 0 = not selected
    - Values in between = partially selected (anti-aliased edges)

    Attributes:
        mask: The selection mask as a grayscale image.
        bounds: Bounding box of the selection (x, y, width, height).
    """

    mask: NDArray[np.uint8]
    bounds: tuple[int, int, int, int] = field(default=(0, 0, 0, 0))

    def __post_init__(self) -> None:
        """Calculate bounds from mask if not provided."""
        if self.bounds == (0, 0, 0, 0) and self.mask is not None:
            self._calculate_bounds()

    def _calculate_bounds(self) -> None:
        """Calculate bounding box from mask."""
        rows = np.any(self.mask > 0, axis=1)
        cols = np.any(self.mask > 0, axis=0)

        if not np.any(rows) or not np.any(cols):
            self.bounds = (0, 0, 0, 0)
            return

        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        self.bounds = (
            int(x_min),
            int(y_min),
            int(x_max - x_min + 1),
            int(y_max - y_min + 1),
        )

    @property
    def is_empty(self) -> bool:
        """Check if the selection is empty."""
        return not np.any(self.mask > 0)

    @property
    def area(self) -> int:
        """Get the area of fully selected pixels."""
        return int(np.sum(self.mask == 255))

    @property
    def area_weighted(self) -> float:
        """Get the area weighted by selection intensity."""
        return float(np.sum(self.mask.astype(np.float32) / 255.0))

    def copy(self) -> Selection:
        """Create a copy of the selection."""
        return Selection(
            mask=self.mask.copy(),
            bounds=self.bounds,
        )

    def invert(self) -> Selection:
        """Return an inverted selection."""
        return Selection(
            mask=255 - self.mask,
        )

    def union(self, other: Selection) -> Selection:
        """Combine two selections using the maximum value."""
        combined = np.maximum(self.mask, other.mask)
        return Selection(mask=combined)

    def intersection(self, other: Selection) -> Selection:
        """Return the intersection of two selections."""
        combined = np.minimum(self.mask, other.mask)
        return Selection(mask=combined)

    def difference(self, other: Selection) -> Selection:
        """Subtract another selection from this one."""
        diff = self.mask.astype(np.int16) - other.mask.astype(np.int16)
        combined = np.clip(diff, 0, 255).astype(np.uint8)
        return Selection(mask=combined)

    def apply_to_image(
        self,
        image: NDArray[np.uint8],
        *,
        fill_color: tuple[int, int, int, int] = (0, 0, 0, 0),
    ) -> NDArray[np.uint8]:
        """Apply selection as alpha to an image.

        Args:
            image: Input image (BGR or BGRA).
            fill_color: Color to fill unselected areas.

        Returns:
            Image with selection applied as alpha.
        """
        import cv2

        # Ensure BGRA
        if image.ndim == 2:
            result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
        elif image.shape[2] == 3:
            result = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        else:
            result = image.copy()

        # Apply selection mask to alpha
        result[:, :, 3] = self.mask

        return result

    def to_binary(self, threshold: int = 128) -> NDArray[np.uint8]:
        """Convert to binary mask.

        Args:
            threshold: Threshold value for binarization.

        Returns:
            Binary mask (0 or 255).
        """
        return np.where(self.mask >= threshold, 255, 0).astype(np.uint8)
