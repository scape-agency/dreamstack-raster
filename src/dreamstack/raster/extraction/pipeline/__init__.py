# -*- coding: utf-8 -*-

"""
Dreamstack Raster - Extraction Pipeline Module
==============================================

High-level pipelines for batch object extraction
with progress tracking and error handling.

"""

from __future__ import annotations

from dreamstack.raster.extraction.pipeline.batch import (
    BatchPipeline,
    BatchResult,
    PipelineConfig,
)
from dreamstack.raster.extraction.pipeline.operations import (
    find_images,
    process_directory,
    process_image,
)

__all__: list[str] = [
    # Pipeline Classes
    "PipelineConfig",
    "BatchPipeline",
    "BatchResult",
    # Operations
    "find_images",
    "process_image",
    "process_directory",
]
