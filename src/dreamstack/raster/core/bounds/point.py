# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Point
=========================

2D point/vector representation.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Point:
    """
    Represents a 2D point or vector.

    Attributes:
        x: X coordinate
        y: Y coordinate
    """

    x: float = 0.0
    y: float = 0.0

    def __add__(self, other: Point) -> Point:
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Point) -> Point:
        return Point(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> Point:
        return Point(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar: float) -> Point:
        return Point(self.x / scalar, self.y / scalar)

    def __neg__(self) -> Point:
        return Point(-self.x, -self.y)

    def __iter__(self) -> Iterator[float]:
        yield self.x
        yield self.y

    @property
    def length(self) -> float:
        """Calculate the length/magnitude of the vector."""
        return math.sqrt(self.x * self.x + self.y * self.y)

    @property
    def normalized(self) -> Point:
        """Return a normalized (unit length) version of this vector."""
        length = self.length
        if length == 0:
            return Point(0, 0)
        return self / length

    def dot(self, other: Point) -> float:
        """Calculate dot product with another point/vector."""
        return self.x * other.x + self.y * other.y

    def cross(self, other: Point) -> float:
        """Calculate 2D cross product (z-component of 3D cross)."""
        return self.x * other.y - self.y * other.x

    def distance_to(self, other: Point) -> float:
        """Calculate distance to another point."""
        return (self - other).length

    def angle_to(self, other: Point) -> float:
        """Calculate angle to another point in radians."""
        return math.atan2(other.y - self.y, other.x - self.x)

    def rotate(self, angle: float, center: Point | None = None) -> Point:
        """Rotate point around a center point by angle (radians)."""
        if center is None:
            center = Point(0, 0)

        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        dx = self.x - center.x
        dy = self.y - center.y

        return Point(
            center.x + dx * cos_a - dy * sin_a,
            center.y + dx * sin_a + dy * cos_a,
        )

    def lerp(self, other: Point, t: float) -> Point:
        """Linear interpolation between this point and another."""
        return Point(
            self.x + (other.x - self.x) * t, self.y + (other.y - self.y) * t
        )

    def to_int(self) -> Point:
        """Convert to integer coordinates."""
        return Point(int(self.x), int(self.y))

    def to_tuple(self) -> tuple[float, float]:
        """Convert to tuple."""
        return (self.x, self.y)

    def to_int_tuple(self) -> tuple[int, int]:
        """Convert to integer tuple."""
        return (int(self.x), int(self.y))

    @classmethod
    def from_tuple(cls, t: tuple[float, float]) -> Point:
        """Create from tuple."""
        return cls(t[0], t[1])

    @classmethod
    def from_polar(cls, radius: float, angle: float) -> Point:
        """Create from polar coordinates."""
        return cls(radius * math.cos(angle), radius * math.sin(angle))
