"""
Cutout Result
=============

Dataclass for a cutout match result.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class CutoutResult:
    """A cutout match result.

    Attributes
    ----------
    path : Path
        Path to cutout image.
    label : str
        Object label.
    confidence : float
        Detection confidence.
    segments : list[Path]
        Paths to segment images.
    source_image : str
        Original source image name.
    metadata_path : Path
        Path to parent metadata.json.
    """

    path: Path
    label: str
    confidence: float
    segments: list[Path]
    source_image: str
    metadata_path: Path

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "path": str(self.path),
            "label": self.label,
            "confidence": self.confidence,
            "segments": [str(p) for p in self.segments],
            "source_image": self.source_image,
        }
