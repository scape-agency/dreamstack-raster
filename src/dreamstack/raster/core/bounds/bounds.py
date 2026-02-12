"""
Dreamstack Raster - Bounds
==========================

Rectangular bounding box representation.

"""

from __future__ import annotations

from dataclasses import dataclass

from dreamstack.raster.core.bounds.point import Point
from dreamstack.raster.core.bounds.size import Size


@dataclass(slots=True)
class Bounds:
    """
    Represents a rectangular bounding box.

    Attributes:
        x: Left edge X coordinate
        y: Top edge Y coordinate
        width: Width of bounds
        height: Height of bounds
    """

    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0

    @property
    def left(self) -> float:
        """Get left edge."""
        return self.x

    @property
    def top(self) -> float:
        """Get top edge."""
        return self.y

    @property
    def right(self) -> float:
        """Get right edge."""
        return self.x + self.width

    @property
    def bottom(self) -> float:
        """Get bottom edge."""
        return self.y + self.height

    @property
    def center(self) -> Point:
        """Get center point."""
        return Point(self.x + self.width / 2, self.y + self.height / 2)

    @property
    def top_left(self) -> Point:
        """Get top-left corner."""
        return Point(self.x, self.y)

    @property
    def top_right(self) -> Point:
        """Get top-right corner."""
        return Point(self.right, self.y)

    @property
    def bottom_left(self) -> Point:
        """Get bottom-left corner."""
        return Point(self.x, self.bottom)

    @property
    def bottom_right(self) -> Point:
        """Get bottom-right corner."""
        return Point(self.right, self.bottom)

    @property
    def size(self) -> Size:
        """Get size."""
        return Size(self.width, self.height)

    @property
    def area(self) -> float:
        """Calculate area."""
        return self.width * self.height

    @property
    def perimeter(self) -> float:
        """Calculate perimeter."""
        return 2 * (self.width + self.height)

    @property
    def is_empty(self) -> bool:
        """Check if bounds has zero area."""
        return self.width <= 0 or self.height <= 0

    def contains_point(self, point: Point) -> bool:
        """Check if point is inside bounds."""
        return self.x <= point.x < self.right and self.y <= point.y < self.bottom

    def contains_bounds(self, other: Bounds) -> bool:
        """Check if this bounds fully contains another."""
        return (
            self.x <= other.x
            and self.y <= other.y
            and self.right >= other.right
            and self.bottom >= other.bottom
        )

    def intersects(self, other: Bounds) -> bool:
        """Check if this bounds intersects with another."""
        return not (
            self.right <= other.x
            or other.right <= self.x
            or self.bottom <= other.y
            or other.bottom <= self.y
        )

    def intersection(self, other: Bounds) -> Bounds | None:
        """Get intersection with another bounds, or None if no intersection."""
        if not self.intersects(other):
            return None

        x = max(self.x, other.x)
        y = max(self.y, other.y)
        right = min(self.right, other.right)
        bottom = min(self.bottom, other.bottom)

        return Bounds(x, y, right - x, bottom - y)

    def union(self, other: Bounds) -> Bounds:
        """Get union (bounding box) of this and another bounds."""
        if self.is_empty:
            return Bounds(other.x, other.y, other.width, other.height)
        if other.is_empty:
            return Bounds(self.x, self.y, self.width, self.height)

        x = min(self.x, other.x)
        y = min(self.y, other.y)
        right = max(self.right, other.right)
        bottom = max(self.bottom, other.bottom)

        return Bounds(x, y, right - x, bottom - y)

    def expand(self, amount: float) -> Bounds:
        """Expand bounds by amount in all directions."""
        return Bounds(
            self.x - amount,
            self.y - amount,
            self.width + amount * 2,
            self.height + amount * 2,
        )

    def contract(self, amount: float) -> Bounds:
        """Contract bounds by amount in all directions."""
        return self.expand(-amount)

    def translate(self, dx: float, dy: float) -> Bounds:
        """Translate bounds by offset."""
        return Bounds(self.x + dx, self.y + dy, self.width, self.height)

    def scale(self, sx: float, sy: float | None = None) -> Bounds:
        """Scale bounds from center."""
        if sy is None:
            sy = sx

        center = self.center
        new_width = self.width * sx
        new_height = self.height * sy

        return Bounds(
            center.x - new_width / 2,
            center.y - new_height / 2,
            new_width,
            new_height,
        )

    def to_int(self) -> Bounds:
        """Convert to integer coordinates."""
        return Bounds(int(self.x), int(self.y), int(self.width), int(self.height))

    def to_tuple(self) -> tuple[float, float, float, float]:
        """Convert to tuple (x, y, width, height)."""
        return (self.x, self.y, self.width, self.height)

    def to_ltrb(self) -> tuple[float, float, float, float]:
        """Convert to left, top, right, bottom tuple."""
        return (self.left, self.top, self.right, self.bottom)

    @classmethod
    def from_points(cls, p1: Point, p2: Point) -> Bounds:
        """Create from two corner points."""
        x = min(p1.x, p2.x)
        y = min(p1.y, p2.y)
        width = abs(p2.x - p1.x)
        height = abs(p2.y - p1.y)
        return cls(x, y, width, height)

    @classmethod
    def from_center(cls, center: Point, size: Size) -> Bounds:
        """Create from center point and size."""
        return cls(
            center.x - size.width / 2,
            center.y - size.height / 2,
            size.width,
            size.height,
        )

    @classmethod
    def from_ltrb(cls, left: float, top: float, right: float, bottom: float) -> Bounds:
        """Create from left, top, right, bottom."""
        return cls(left, top, right - left, bottom - top)

    @classmethod
    def from_size(cls, size: Size) -> Bounds:
        """Create from size at origin."""
        return cls(0, 0, size.width, size.height)
