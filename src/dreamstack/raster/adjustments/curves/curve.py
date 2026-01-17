# -*- coding: utf-8 -*-

"""Curve dataclass for curves adjustment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
from scipy import interpolate

from dreamstack.raster.adjustments.curves.curve_point import CurvePoint


@dataclass
class Curve:
    """A curves adjustment curve."""

    points: List[CurvePoint] = field(
        default_factory=lambda: [CurvePoint(0, 0), CurvePoint(255, 255)]
    )

    def __post_init__(self):
        # Ensure sorted by input
        self.points = sorted(self.points, key=lambda p: p.input)

        # Ensure endpoints
        if not self.points or self.points[0].input != 0:
            self.points.insert(0, CurvePoint(0, 0))
        if self.points[-1].input != 255:
            self.points.append(CurvePoint(255, 255))

    def add_point(self, input_val: float, output_val: float) -> None:
        """Add a point to the curve."""
        # Remove existing point at same input
        self.points = [
            p for p in self.points if abs(p.input - input_val) > 0.5
        ]
        self.points.append(CurvePoint(input_val, output_val))
        self.points = sorted(self.points, key=lambda p: p.input)

    def remove_point(self, input_val: float) -> None:
        """Remove a point from the curve."""
        # Don't remove endpoints
        if input_val <= 1 or input_val >= 254:
            return
        self.points = [
            p for p in self.points if abs(p.input - input_val) > 0.5
        ]

    def get_lookup_table(self, size: int = 256) -> np.ndarray:
        """Generate lookup table from curve."""
        x = np.array([p.input for p in self.points])
        y = np.array([p.output for p in self.points])

        if len(x) < 2:
            return np.arange(size)

        if len(x) == 2:
            # Linear interpolation
            lut = np.interp(
                np.arange(size), x * (size - 1) / 255, y * (size - 1) / 255
            )
        else:
            # Cubic spline interpolation
            try:
                spline = interpolate.CubicSpline(x, y, bc_type="clamped")
                lut = spline(np.linspace(0, 255, size))
            except Exception:
                # Fallback to linear
                lut = np.interp(
                    np.arange(size), x * (size - 1) / 255, y * (size - 1) / 255
                )

        return np.clip(lut * (size - 1) / 255, 0, size - 1)
