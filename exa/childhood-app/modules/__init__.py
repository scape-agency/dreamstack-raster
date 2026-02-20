"""
Childhood App Modules
=====================

Art installation image preprocessing modules.
"""

from modules.config import AppConfig, CutoutConfig, SegmentConfig, CanvasConfig
from modules.grid import segment_image, GridSegment
from modules.effects import apply_effects, EffectType

__all__ = [
    "AppConfig",
    "CutoutConfig",
    "SegmentConfig",
    "CanvasConfig",
    "segment_image",
    "GridSegment",
    "apply_effects",
    "EffectType",
]
