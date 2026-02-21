# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Detector Factory
================

Factory function for creating detection backends.
"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.detection.base import BaseDetector

    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.detection.config import DetectionConfig


def create_detector(config: DetectionConfig) -> BaseDetector:
    """Create a detector instance based on configuration.

    Parameters
    ----------
    config : DetectionConfig
        Detection configuration with backend selection.

    Returns
    -------
    BaseDetector
        Configured detector instance.

    Raises
    ------
    ValueError
        If backend is not supported.
    ImportError
        If required dependencies are not installed.

    Example
    -------
    >>> from dreamstack.raster.detection import create_detector, DetectionConfig
    >>>
    >>> config = DetectionConfig(backend="ultralytics")
    >>> detector = create_detector(config)
    >>> results = detector.detect(image)
    """
    if config.backend == "ultralytics":
        # pylint: disable=import-outside-toplevel
        from dreamstack.raster.detection.backends.ultralytics import (
            UltralyticsDetector,
        )

        return UltralyticsDetector(config)

    elif config.backend == "grounding_dino_sam":
        # pylint: disable=import-outside-toplevel
        from dreamstack.raster.detection.backends.grounding_dino_sam import (
            GroundingDinoSamDetector,
        )

        detector = GroundingDinoSamDetector(config)
        if config.text_prompts:
            detector.set_prompts(config.text_prompts)
        return detector

    elif config.backend == "transformers":
        # Placeholder for future HuggingFace DETR backend
        raise NotImplementedError(
            "transformers backend not yet implemented. "
            "Use 'ultralytics' or 'grounding_dino_sam' backend instead."
        )

    else:
        raise ValueError(
            f"Unknown detection backend: {config.backend}. "
            f"Supported backends: 'ultralytics', 'transformers'"
        )
