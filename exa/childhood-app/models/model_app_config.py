"""
Application Configuration
=========================

Main application configuration combining all config dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from models.model_cutout_config import CutoutConfig
from models.model_segment_config import SegmentConfig
from models.model_canvas_config import CanvasConfig
from models.model_effect_config import EffectConfig


@dataclass
class AppConfig:
    """Main application configuration.

    Attributes
    ----------
    cutout : CutoutConfig
        Cutout configuration.
    segment : SegmentConfig
        Segmentation configuration.
    canvas : CanvasConfig
        Canvas configuration.
    effects : EffectConfig
        Effects configuration.
    vision_backend : Literal["openai", "mistral"]
        AI vision backend for description. Default openai.
    detection_backend : Literal["ultralytics", "grounding_dino_sam"]
        Detection backend. Default ultralytics.
    confidence_threshold : float
        Detection confidence threshold. Default 0.5.
    """

    cutout: CutoutConfig = field(default_factory=CutoutConfig)
    segment: SegmentConfig = field(default_factory=SegmentConfig)
    canvas: CanvasConfig = field(default_factory=CanvasConfig)
    effects: EffectConfig = field(default_factory=EffectConfig)
    vision_backend: Literal["openai", "mistral"] = "openai"
    detection_backend: Literal["ultralytics", "grounding_dino_sam"] = (
        "ultralytics"
    )
    confidence_threshold: float = 0.5

    @classmethod
    def default(cls) -> AppConfig:
        """Create default configuration."""
        return cls()

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "cutout": {
                "max_size": self.cutout.max_size,
                "margin": self.cutout.margin,
                "segment_align": self.cutout.segment_align,
            },
            "segment": {
                "segment_size": list(self.segment.segment_size),
                "randomize_offset": self.segment.randomize_offset,
                "max_offset": self.segment.max_offset,
            },
            "canvas": {
                "size": list(self.canvas.size),
                "origin": self.canvas.origin,
            },
            "effects": {
                "drop_shadow": self.effects.drop_shadow,
                "shadow_offset": list(self.effects.shadow_offset),
                "shadow_blur": self.effects.shadow_blur,
                "filters": self.effects.filters,
            },
            "vision_backend": self.vision_backend,
            "detection_backend": self.detection_backend,
            "confidence_threshold": self.confidence_threshold,
        }
