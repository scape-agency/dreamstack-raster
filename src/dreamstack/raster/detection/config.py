# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Detection Config
================

Configuration dataclasses for object detection.
"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Backend = Literal["ultralytics", "transformers", "grounding_dino_sam"]
DeviceType = Literal["auto", "cpu", "cuda", "mps"]
VisionBackend = Literal["openai", "mistral"]


@dataclass
class SegmentationConfig:
    """Configuration for instance segmentation.

    Attributes
    ----------
    enabled : bool
        Whether to perform segmentation (vs bounding box only).
    feather_edges : int
        Feathering radius for mask edges. Default 0.
    min_mask_area : int
        Minimum mask area in pixels. Default 100.
    """

    enabled: bool = True
    feather_edges: int = 0
    min_mask_area: int = 100


@dataclass
class DetectionConfig:
    """Configuration for object detection.

    Attributes
    ----------
    backend : Backend
        Detection backend to use. Default "ultralytics".
    model_name : str
        Model name/path. Default "yolov8n-seg" (nano with segmentation).
    device : DeviceType
        Device for inference. "auto" detects MPS/CUDA. Default "auto".
    confidence_threshold : float
        Minimum confidence for detections. Default 0.5.
    iou_threshold : float
        IoU threshold for NMS. Default 0.5.
    max_detections : int
        Maximum detections per image. Default 100.
    classes : list[int] | None
        Specific class IDs to detect. None = all classes.
    margin : int
        Margin around extracted objects in pixels. Default 10.
    min_dimension : int
        Minimum width/height for valid objects. Default 24.
    segmentation : SegmentationConfig
        Segmentation configuration.

    Examples
    --------
    >>> config = DetectionConfig(
    ...     model_name="yolov8n-seg",
    ...     confidence_threshold=0.6,
    ... )
    >>> pipeline = DetectionPipeline(config)
    """

    backend: Backend = "ultralytics"
    model_name: str = "yolov8n-seg"
    device: DeviceType = "auto"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.5
    max_detections: int = 100
    classes: list[int] | None = None
    margin: int = 10
    # Text prompts for open-vocabulary detection (Grounding DINO)
    text_prompts: list[str] | None = None
    # AI vision for auto-generating prompts
    use_ai_description: bool = False
    vision_backend: VisionBackend = "openai"
    min_dimension: int = 24
    segmentation: SegmentationConfig = field(
        default_factory=SegmentationConfig
    )
