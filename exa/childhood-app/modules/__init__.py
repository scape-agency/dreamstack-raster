"""
Childhood App Modules
=====================

Art installation image preprocessing modules.

This module provides backwards-compatible imports from the old structure.
New code should import directly from models/, services/, and utils/.
"""

# Backwards-compatible imports from new structure
from models.model_app_config import AppConfig
from models.model_cutout_config import CutoutConfig
from models.model_segment_config import SegmentConfig
from models.model_canvas_config import CanvasConfig
from models.model_effect_config import EffectConfig
from models.model_effect_type import EffectType
from models.model_effect_result import EffectResult
from models.model_grid_segment import GridSegment

from services.service_segment_image import segment_image
from services.service_apply_effects import apply_effects

__all__: list[str] = [

    # Config models
    "AppConfig",
    "CutoutConfig",
    "SegmentConfig",
    "CanvasConfig",
    "EffectConfig",
    # Effect models
    "EffectType",
    "EffectResult",
    # Grid models
    "GridSegment",
    # Services
    "segment_image",
    "apply_effects",
]
