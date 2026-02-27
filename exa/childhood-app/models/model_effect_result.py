"""
Effect Result
=============

Result dataclass from applying effects.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image


@dataclass
class EffectResult:
    """Result of applying effects.

    Attributes
    ----------
    image : Image.Image
        Processed image.
    effects_applied : list[str]
        Names of effects applied.
    """

    image: Image.Image
    effects_applied: list[str]
