"""
Gradient Drawing
================

Gradient generation operations.

"""

from __future__ import annotations

from dreamstack.raster.drawing.gradient.angular_gradient import (
    angular_gradient,
)
from dreamstack.raster.drawing.gradient.diamond_gradient import (
    diamond_gradient,
)
from dreamstack.raster.drawing.gradient.gradient_stop import GradientStop
from dreamstack.raster.drawing.gradient.linear_gradient import linear_gradient
from dreamstack.raster.drawing.gradient.radial_gradient import radial_gradient

__all__: list[str] = [
    "GradientStop",
    "linear_gradient",
    "radial_gradient",
    "angular_gradient",
    "diamond_gradient",
]
