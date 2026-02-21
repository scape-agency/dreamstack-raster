# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Text Style
==========

Text styling configuration.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class FontWeight(Enum):
    """Font weight options."""

    NORMAL = 0
    BOLD = 1
    ITALIC = 2
    BOLD_ITALIC = 3


@dataclass
class TextStyle:
    """Configuration for text rendering.

    Attributes:
        font_scale: Font size scaling factor.
        color: Text color (RGBA).
        thickness: Text stroke thickness.
        font_weight: Font weight style.
        line_spacing: Line spacing multiplier.
        background_color: Optional background color (RGBA).
        padding: Padding around text when background is used.
    """

    font_scale: float = 1.0
    color: tuple[int, int, int, int] = field(default=(0, 0, 0, 255))
    thickness: int = 1
    font_weight: FontWeight = FontWeight.NORMAL
    line_spacing: float = 1.5
    background_color: tuple[int, int, int, int] | None = None
    padding: int = 5

    def with_color(self, color: tuple[int, int, int, int]) -> TextStyle:
        """Return a copy with new color."""
        return TextStyle(
            font_scale=self.font_scale,
            color=color,
            thickness=self.thickness,
            font_weight=self.font_weight,
            line_spacing=self.line_spacing,
            background_color=self.background_color,
            padding=self.padding,
        )

    def with_size(self, scale: float) -> TextStyle:
        """Return a copy with new font scale."""
        return TextStyle(
            font_scale=scale,
            color=self.color,
            thickness=self.thickness,
            font_weight=self.font_weight,
            line_spacing=self.line_spacing,
            background_color=self.background_color,
            padding=self.padding,
        )
