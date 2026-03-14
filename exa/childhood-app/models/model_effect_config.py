"""
Effect Configuration
====================

Configuration dataclass for image effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EffectConfig:
    """Configuration for image effects.

    Attributes
    ----------
    drop_shadow : bool
        Apply drop shadow. Default True.
    shadow_offset : tuple[int, int]
        Shadow offset (x, y). Default (5, 5).
    shadow_blur : int
        Shadow blur radius. Default 10.
    shadow_opacity : float
        Shadow opacity (0-1). Default 0.5.
    filters : list[str]
        List of filter names to apply. Default empty.
    """

    drop_shadow: bool = True
    shadow_offset: tuple[int, int] = (5, 5)
    shadow_blur: int = 10
    shadow_opacity: float = 0.5
    filters: list[str] = field(default_factory=list)
