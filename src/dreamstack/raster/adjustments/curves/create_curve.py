# -*- coding: utf-8 -*-

"""Create curve function."""

from __future__ import annotations

from typing import List, Tuple

from dreamstack.raster.adjustments.curves.curve import Curve
from dreamstack.raster.adjustments.curves.curve_point import CurvePoint


def create_curve(points: List[Tuple[float, float]]) -> Curve:
    """
    Create a curve from a list of (input, output) tuples.

    Args:
        points: List of (input, output) tuples

    Returns:
        Curve object
    """
    curve_points = [CurvePoint(p[0], p[1]) for p in points]
    return Curve(points=curve_points)
