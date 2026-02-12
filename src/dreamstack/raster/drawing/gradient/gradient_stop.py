"""
Gradient Stop
=============

Color stop definition for gradients.

"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GradientStop:
    """A color stop in a gradient.

    Attributes:
        position: Position along gradient (0.0 to 1.0).
        color: RGBA color at this position.
    """

    position: float
    color: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        """Validate position."""
        self.position = max(0.0, min(1.0, self.position))
