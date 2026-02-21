# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Detection Backends
==================

Object detection backend implementations.
"""

from dreamstack.raster.detection.backends.ultralytics import (
    UltralyticsDetector,
)


# Lazy import for Grounding DINO + SAM (heavy dependencies)
def get_grounding_dino_sam_detector():
    """Get GroundingDinoSamDetector class (lazy import)."""
    from dreamstack.raster.detection.backends.grounding_dino_sam import (
        GroundingDinoSamDetector,
    )

    return GroundingDinoSamDetector


__all__ = [
    "UltralyticsDetector",
    "get_grounding_dino_sam_detector",
]
