# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Curve point dataclass."""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
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
