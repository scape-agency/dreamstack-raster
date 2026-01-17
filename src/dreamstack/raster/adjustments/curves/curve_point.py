# -*- coding: utf-8 -*-

"""Curve point dataclass."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CurvePoint:
    """A point on a curve."""

    input: float  # 0-255
    output: float  # 0-255

    def __post_init__(self):
        self.input = max(0, min(255, self.input))
        self.output = max(0, min(255, self.output))
