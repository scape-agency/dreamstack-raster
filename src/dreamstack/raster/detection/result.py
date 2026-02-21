# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Detection Result
================

Data classes for detection results.
"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class DetectionResult:
    """Single object detection result.

    Attributes
    ----------
    label : str
        Class label (e.g., "dog", "person", "car").
    class_id : int
        Numeric class ID from the model.
    confidence : float
        Detection confidence score (0-1).
    bbox : tuple[int, int, int, int]
        Bounding box as (x, y, width, height).
    mask : NDArray[np.uint8] | None
        Segmentation mask (same size as bbox region), or None if no segmentation.
    """

    label: str
    class_id: int
    confidence: float
    bbox: tuple[int, int, int, int]
    mask: NDArray[np.uint8] | None = None

    @property
    def x(self) -> int:
        """Bounding box x coordinate."""
        return self.bbox[0]

    @property
    def y(self) -> int:
        """Bounding box y coordinate."""
        return self.bbox[1]

    @property
    def width(self) -> int:
        """Bounding box width."""
        return self.bbox[2]

    @property
    def height(self) -> int:
        """Bounding box height."""
        return self.bbox[3]

    @property
    def area(self) -> int:
        """Bounding box area in pixels."""
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        """Center point of bounding box."""
        return (self.x + self.width / 2, self.y + self.height / 2)

    @property
    def has_mask(self) -> bool:
        """Whether segmentation mask is available."""
        return self.mask is not None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "label": self.label,
            "class_id": self.class_id,
            "confidence": round(self.confidence, 4),
            "bbox": list(self.bbox),
            "has_mask": self.has_mask,
        }


@dataclass
class ImageDetectionResult:
    """Detection results for a single image.

    Attributes
    ----------
    source_path : Path | None
        Path to source image.
    image_size : tuple[int, int]
        Image dimensions as (height, width).
    detections : list[DetectionResult]
        List of detected objects.
    """

    source_path: Path | None
    image_size: tuple[int, int]
    detections: list[DetectionResult] = field(default_factory=list)

    @property
    def num_detections(self) -> int:
        """Number of detections."""
        return len(self.detections)

    @property
    def labels(self) -> list[str]:
        """List of unique labels detected."""
        return list(set(d.label for d in self.detections))

    def get_by_label(self, label: str) -> list[DetectionResult]:
        """Get all detections with a specific label."""
        return [d for d in self.detections if d.label == label]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "source_image": (
                str(self.source_path) if self.source_path else None
            ),
            "image_size": {
                "height": self.image_size[0],
                "width": self.image_size[1],
            },
            "num_detections": self.num_detections,
            "labels": self.labels,
            "detections": [d.to_dict() for d in self.detections],
        }
