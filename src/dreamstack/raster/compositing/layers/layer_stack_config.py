"""Layer stack configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LayerStackConfig:
    """Configuration for layer stacking.

    Attributes:
        output_format: Output image format.
        resize_to_base: Resize all layers to match base layer.
        fill_color: Background fill color (RGBA).
    """

    output_format: str = "png"
    resize_to_base: bool = True
    fill_color: tuple[int, int, int, int] = (0, 0, 0, 0)
