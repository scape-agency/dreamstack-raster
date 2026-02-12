"""
Brush Preset
============

Preset brush configurations.

"""

from __future__ import annotations

from enum import Enum


class BrushPreset(Enum):
    """Predefined brush presets.

    Attributes:
        SOFT_ROUND: Soft-edged circular brush.
        HARD_ROUND: Hard-edged circular brush.
        AIRBRUSH: Very soft, diffuse brush.
        PENCIL: Hard, 1px pencil-like brush.
        CHALK: Textured chalk-like brush.
        WATERCOLOR: Wet-edge watercolor effect.
        ERASER_SOFT: Soft-edged eraser.
        ERASER_HARD: Hard-edged eraser.
    """

    SOFT_ROUND = "soft_round"
    HARD_ROUND = "hard_round"
    AIRBRUSH = "airbrush"
    PENCIL = "pencil"
    CHALK = "chalk"
    WATERCOLOR = "watercolor"
    ERASER_SOFT = "eraser_soft"
    ERASER_HARD = "eraser_hard"
