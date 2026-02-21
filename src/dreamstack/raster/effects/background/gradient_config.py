# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Gradient Configuration
======================

Configuration dataclass for gradient backgrounds.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

GradientDirection = Literal["horizontal", "vertical", "diagonal", "radial"]


@dataclass
class GradientConfig:
    """Configuration for gradient backgrounds.

    Attributes:
        start_color: Starting RGB color.
        end_color: Ending RGB color.
        direction: Gradient direction.
        center: Center point for radial gradients (normalized 0-1).
    """

    start_color: tuple[int, int, int] = (255, 255, 255)
    end_color: tuple[int, int, int] = (200, 200, 200)
    direction: GradientDirection = "vertical"
    center: tuple[float, float] = (0.5, 0.5)
