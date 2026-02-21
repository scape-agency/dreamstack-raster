# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Bevel Style
===========

Bevel and emboss style configuration.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class BevelType(Enum):
    """Type of bevel effect."""

    OUTER_BEVEL = "outer_bevel"
    INNER_BEVEL = "inner_bevel"
    EMBOSS = "emboss"
    PILLOW_EMBOSS = "pillow_emboss"


class BevelTechnique(Enum):
    """Bevel rendering technique."""

    SMOOTH = "smooth"
    CHISEL_HARD = "chisel_hard"
    CHISEL_SOFT = "chisel_soft"


@dataclass
class BevelStyle:
    """Configuration for bevel and emboss effects.

    Attributes:
        bevel_type: Type of bevel effect.
        technique: Rendering technique.
        depth: Bevel depth (1-100).
        size: Bevel size in pixels.
        soften: Soften amount in pixels.
        angle: Light angle in degrees.
        altitude: Light altitude in degrees.
        highlight_color: Highlight color (RGB).
        highlight_opacity: Highlight opacity (0.0 to 1.0).
        shadow_color: Shadow color (RGB).
        shadow_opacity: Shadow opacity (0.0 to 1.0).
    """

    bevel_type: BevelType = BevelType.INNER_BEVEL
    technique: BevelTechnique = BevelTechnique.SMOOTH
    depth: int = 50
    size: int = 5
    soften: int = 0
    angle: float = 120.0
    altitude: float = 30.0
    highlight_color: tuple[int, int, int] = (255, 255, 255)
    highlight_opacity: float = 0.75
    shadow_color: tuple[int, int, int] = (0, 0, 0)
    shadow_opacity: float = 0.75
