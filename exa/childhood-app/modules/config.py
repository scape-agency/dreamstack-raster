"""
Configuration
=============

Configuration dataclasses for the childhood app preprocessor.
"""

# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class CutoutConfig:
    """Configuration for bounding box cutouts.

    Attributes
    ----------
    max_size : int
        Maximum size for largest dimension. Default 1200.
    margin : int
        Margin around bounding box in pixels. Default 50.
    segment_align : bool
        Align smallest dimension to segment size multiple. Default True.
    background_color : tuple[int, int, int, int]
        RGBA background color for padding. Default transparent.
    """

    max_size: int = 1200
    margin: int = 50
    segment_align: bool = True
    background_color: tuple[int, int, int, int] = (0, 0, 0, 0)


@dataclass
class SegmentConfig:
    """Configuration for grid segmentation.

    Attributes
    ----------
    segment_size : tuple[int, int]
        Target segment size (width, height). Default 400x300.
    randomize_offset : bool
        Apply random offset to grid lines. Default True.
    max_offset : int
        Maximum random offset in pixels (equals margin). Default 50.
    empty_alpha_threshold : int
        Alpha value at or below which pixel is considered empty (padded).
        Only detects pixels we added as transparent padding. Default 0.
    generate_inbetweens : bool
        Generate horizontal and vertical in-between segments at half-positions.
        Creates ~2x more segments. Default False.
    """

    segment_size: tuple[int, int] = (400, 300)
    randomize_offset: bool = True
    max_offset: int = 50
    empty_alpha_threshold: int = 0
    generate_inbetweens: bool = False


@dataclass
class CanvasConfig:
    """Configuration for the art installation canvas.

    Attributes
    ----------
    size : tuple[int, int]
        Canvas size (width, height). Default 8000x3000.
    background_color : tuple[int, int, int, int]
        RGBA background color. Default transparent.
    origin : Literal["bottom-left", "top-left"]
        Coordinate origin for placement. Default bottom-left.
    """

    size: tuple[int, int] = (8000, 3000)
    background_color: tuple[int, int, int, int] = (255, 255, 255, 255)
    origin: Literal["bottom-left", "top-left"] = "bottom-left"


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
