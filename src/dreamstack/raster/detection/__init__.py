# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Detection Module
================

Object detection and extraction with multiple backend support.

This module provides:
- Multi-backend object detection (YOLO, HuggingFace)
- Instance segmentation for precise cutouts
- Batch processing pipeline
- JSON metadata generation

Quick Start
-----------
>>> from dreamstack.raster.detection import DetectionPipeline, DetectionConfig
>>>
>>> config = DetectionConfig(
...     model_name="yolov8n-seg",  # Nano model, fastest
...     confidence_threshold=0.5,
... )
>>>
>>> pipeline = DetectionPipeline(config)
>>> results = pipeline.process_directory("./images", "./output")
>>> print(f"Extracted {results.total_objects} objects")

Output Structure
----------------
For each image, creates a folder with:
- metadata.json: Image description and object list
- {label}_{n}.png: Extracted objects with alpha channel

Example output:
    output/
    ├── photo_001/
    │   ├── metadata.json
    │   ├── dog_1.png
    │   └── person_1.png
    └── ...

Backends
--------
- ultralytics (default): YOLOv8/v11 with instance segmentation
- transformers: HuggingFace DETR (future)

Requirements
------------
pip install ultralytics  # For YOLO backend
"""

from dreamstack.raster.detection.base import BaseDetector
from dreamstack.raster.detection.config import (
    Backend,
    DetectionConfig,
    DeviceType,
    SegmentationConfig,
)
from dreamstack.raster.detection.describer import (
    DescriptionResult,
    ImageDescriber,
)
from dreamstack.raster.detection.extractor import (
    DetectionExtractor,
    ExtractedDetection,
)
from dreamstack.raster.detection.factory import create_detector
from dreamstack.raster.detection.metadata import (
    ImageMetadata,
    ObjectMetadata,
    create_image_metadata,
    generate_description,
    load_metadata,
    save_metadata,
)
from dreamstack.raster.detection.pipeline import (
    DetectionPipeline,
    PipelineResult,
)
from dreamstack.raster.detection.result import (
    DetectionResult,
    ImageDetectionResult,
)

__all__: list[str] = [
    # Config
    "DetectionConfig",
    "SegmentationConfig",
    "Backend",
    "DeviceType",
    # Results
    "DetectionResult",
    "ImageDetectionResult",
    # Base
    "BaseDetector",
    # Factory
    "create_detector",
    # Extractor
    "DetectionExtractor",
    "ExtractedDetection",
    # Pipeline
    "DetectionPipeline",
    "PipelineResult",
    # Metadata
    "ImageMetadata",
    "ObjectMetadata",
    "create_image_metadata",
    "save_metadata",
    "load_metadata",
    "generate_description",
    # Description
    "ImageDescriber",
    "DescriptionResult",
]
