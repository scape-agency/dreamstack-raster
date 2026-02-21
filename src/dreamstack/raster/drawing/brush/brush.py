# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Brush
=====

Brush configuration for drawing operations.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from dreamstack.raster.drawing.brush.brush_preset import BrushPreset


@dataclass
class Brush:
    """Configurable brush for drawing operations.

    Attributes:
        size: Brush diameter in pixels.
        hardness: Edge hardness (0.0 = soft, 1.0 = hard).
        opacity: Brush opacity (0.0 = transparent, 1.0 = opaque).
        flow: Paint flow rate (0.0 = none, 1.0 = full).
        spacing: Spacing between dabs as fraction of size.
        angle: Brush rotation in degrees.
        roundness: Brush roundness (0.0 = flat, 1.0 = circular).
        color: RGBA brush color.
    """

    size: int = 20
    hardness: float = 0.5
    opacity: float = 1.0
    flow: float = 1.0
    spacing: float = 0.25
    angle: float = 0.0
    roundness: float = 1.0
    color: tuple[int, int, int, int] = field(default=(0, 0, 0, 255))

    def __post_init__(self) -> None:
        """Validate and clamp values."""
        self.size = max(1, self.size)
        self.hardness = max(0.0, min(1.0, self.hardness))
        self.opacity = max(0.0, min(1.0, self.opacity))
        self.flow = max(0.0, min(1.0, self.flow))
        self.spacing = max(0.01, min(2.0, self.spacing))
        self.roundness = max(0.01, min(1.0, self.roundness))

    @classmethod
    def from_preset(cls, preset: BrushPreset, size: int = 20) -> Brush:
        """Create a brush from a preset.

        Args:
            preset: The brush preset to use.
            size: Brush size in pixels.

        Returns:
            Configured Brush instance.
        """
        presets = {
            BrushPreset.SOFT_ROUND: {
                "hardness": 0.0,
                "opacity": 1.0,
                "flow": 1.0,
                "spacing": 0.1,
            },
            BrushPreset.HARD_ROUND: {
                "hardness": 1.0,
                "opacity": 1.0,
                "flow": 1.0,
                "spacing": 0.1,
            },
            BrushPreset.AIRBRUSH: {
                "hardness": 0.0,
                "opacity": 0.3,
                "flow": 0.3,
                "spacing": 0.05,
            },
            BrushPreset.PENCIL: {
                "hardness": 1.0,
                "opacity": 1.0,
                "flow": 1.0,
                "spacing": 0.01,
            },
            BrushPreset.CHALK: {
                "hardness": 0.8,
                "opacity": 0.7,
                "flow": 0.8,
                "spacing": 0.2,
            },
            BrushPreset.WATERCOLOR: {
                "hardness": 0.0,
                "opacity": 0.5,
                "flow": 0.4,
                "spacing": 0.1,
            },
            BrushPreset.ERASER_SOFT: {
                "hardness": 0.0,
                "opacity": 1.0,
                "flow": 1.0,
                "spacing": 0.1,
            },
            BrushPreset.ERASER_HARD: {
                "hardness": 1.0,
                "opacity": 1.0,
                "flow": 1.0,
                "spacing": 0.1,
            },
        }

        config = presets.get(preset, {})
        return cls(size=size, **config)

    def create_tip(self) -> NDArray[np.float32]:
        """Create the brush tip mask.

        Returns:
            2D array representing brush tip intensity.
        """
        # Create coordinate grid
        y, x = np.ogrid[
            -self.size // 2 : self.size // 2 + 1,
            -self.size // 2 : self.size // 2 + 1,
        ]

        # Apply roundness (ellipse)
        x_scaled = x.astype(np.float32)
        y_scaled = y.astype(np.float32) / self.roundness

        # Apply rotation
        if self.angle != 0:
            rad = np.radians(self.angle)
            cos_a, sin_a = np.cos(rad), np.sin(rad)
            x_rot = x_scaled * cos_a - y_scaled * sin_a
            y_rot = x_scaled * sin_a + y_scaled * cos_a
            x_scaled, y_scaled = x_rot, y_rot

        # Calculate distance from center
        distance = np.sqrt(x_scaled**2 + y_scaled**2)
        radius = self.size / 2

        # Create falloff based on hardness
        if self.hardness >= 1.0:
            # Hard edge
            tip = (distance <= radius).astype(np.float32)
        else:
            # Soft edge with falloff
            inner_radius = radius * self.hardness
            outer_radius = radius

            tip = np.ones_like(distance, dtype=np.float32)
            mask = distance > inner_radius
            falloff = 1.0 - (distance[mask] - inner_radius) / (
                outer_radius - inner_radius + 1e-6
            )
            tip[mask] = np.clip(falloff, 0, 1)
            tip[distance > outer_radius] = 0

        return tip * self.flow
