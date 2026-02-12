"""
Dreamstack Raster - Size
========================

2D size representation.

"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dreamstack.raster.core.bounds.point import Point


@dataclass(frozen=True, slots=True)
class Size:
    """
    Represents a 2D size.

    Attributes:
        width: Width dimension
        height: Height dimension
    """

    width: float = 0.0
    height: float = 0.0

    def __mul__(self, scalar: float) -> Size:
        return Size(self.width * scalar, self.height * scalar)

    def __truediv__(self, scalar: float) -> Size:
        return Size(self.width / scalar, self.height / scalar)

    def __iter__(self) -> Iterator[float]:
        yield self.width
        yield self.height

    @property
    def area(self) -> float:
        """Calculate area."""
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        """Calculate aspect ratio (width / height)."""
        if self.height == 0:
            return 0
        return self.width / self.height

    @property
    def diagonal(self) -> float:
        """Calculate diagonal length."""
        return math.sqrt(self.width**2 + self.height**2)

    def contains(self, point: Point) -> bool:
        """Check if point is within bounds (0, 0) to (width, height)."""
        return 0 <= point.x < self.width and 0 <= point.y < self.height

    def fit_within(self, max_size: Size, maintain_aspect: bool = True) -> Size:
        """
        Calculate size that fits within max_size.

        Args:
            max_size: Maximum allowed size
            maintain_aspect: Whether to maintain aspect ratio

        Returns:
            New Size that fits within max_size
        """
        if not maintain_aspect:
            return Size(
                min(self.width, max_size.width),
                min(self.height, max_size.height),
            )

        scale = min(
            max_size.width / self.width if self.width > 0 else float("inf"),
            max_size.height / self.height if self.height > 0 else float("inf"),
        )

        if scale >= 1:
            return self

        return self * scale

    def to_int(self) -> Size:
        """Convert to integer dimensions."""
        return Size(int(self.width), int(self.height))

    def to_tuple(self) -> tuple[float, float]:
        """Convert to tuple."""
        return (self.width, self.height)

    def to_int_tuple(self) -> tuple[int, int]:
        """Convert to integer tuple."""
        return (int(self.width), int(self.height))

    @classmethod
    def from_tuple(cls, t: tuple[float, float]) -> Size:
        """Create from tuple."""
        return cls(t[0], t[1])
