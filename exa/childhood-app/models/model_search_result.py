"""
Search Result
=============

Dataclass for a search match result.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from models.model_cutout_result import CutoutResult


@dataclass
class SearchResult:
    """A search match result.

    Attributes
    ----------
    source_image : str
        Original image filename.
    output_dir : Path
        Output directory for this image.
    metadata_path : Path
        Path to metadata.json.
    ai_description : str
        AI-generated description.
    detected_objects : list[str]
        List of detected object types.
    cutouts : list[CutoutResult]
        Available cutouts.
    score : float
        Relevance score (higher is better).
    """

    source_image: str
    output_dir: Path
    metadata_path: Path
    ai_description: str
    detected_objects: list[str]
    cutouts: list[CutoutResult]
    score: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "source_image": self.source_image,
            "output_dir": str(self.output_dir),
            "ai_description": self.ai_description,
            "detected_objects": self.detected_objects,
            "cutouts": [c.to_dict() for c in self.cutouts],
            "score": self.score,
        }
