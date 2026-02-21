# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Detection Metadata
==================

JSON metadata generation for detection results.
"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

# =============================================================================
# Type Checking Imports
# =============================================================================

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.detection.extractor import ExtractedDetection

    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.detection.result import ImageDetectionResult


@dataclass
class ObjectMetadata:
    """Metadata for a single extracted object.

    Attributes
    ----------
    label : str
        Object class label.
    confidence : float
        Detection confidence score.
    bbox : list[int]
        Bounding box [x, y, width, height].
    file : str
        Filename of extracted image.
    """

    label: str
    confidence: float
    bbox: list[int]
    file: str


@dataclass
class ImageMetadata:
    """Metadata for a processed image.

    Attributes
    ----------
    source_image : str
        Original image filename.
    processed_at : str
        ISO timestamp of processing.
    description : str
        Human-readable description of image contents.
    image_size : dict[str, int]
        Image dimensions.
    num_objects : int
        Number of objects detected.
    labels : list[str]
        Unique labels found.
    objects : list[ObjectMetadata]
        Detailed object information.
    ai_description : str | None
        AI-generated description of the image (if available).
    """

    source_image: str
    processed_at: str
    description: str
    image_size: dict[str, int]
    num_objects: int
    labels: list[str]
    objects: list[ObjectMetadata] = field(default_factory=list)
    ai_description: str | None = None


def generate_description(labels: list[str]) -> str:
    """Generate a human-readable description from labels.

    Parameters
    ----------
    labels : list[str]
        List of detected labels (may have duplicates).

    Returns
    -------
    str
        Description like "Image contains 3 objects: dog, bicycle, person"
    """
    if not labels:
        return "No objects detected"

    # Count occurrences
    label_counts: dict[str, int] = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1

    # Build description parts
    parts: list[str] = []
    for label, count in sorted(label_counts.items()):
        if count > 1:
            parts.append(f"{count} {label}s")
        else:
            parts.append(label)

    total = len(labels)
    obj_word = "object" if total == 1 else "objects"

    if len(parts) <= 3:
        items = ", ".join(parts)
    else:
        items = ", ".join(parts[:3]) + f", and {len(parts) - 3} more"

    return f"Image contains {total} {obj_word}: {items}"


def create_image_metadata(
    source_path: Path,
    image_size: tuple[int, int],
    extractions: list[ExtractedDetection],
    ai_description: str | None = None,
) -> ImageMetadata:
    """Create metadata for a processed image.

    Parameters
    ----------
    source_path : Path
        Path to source image.
    image_size : tuple[int, int]
        Image dimensions (height, width).
    extractions : list[ExtractedDetection]
        Extracted objects.
    ai_description : str | None
        AI-generated description of the image.

    Returns
    -------
    ImageMetadata
        Complete metadata for the image.
    """
    labels = [ext.label for ext in extractions]

    objects = [
        ObjectMetadata(
            label=ext.label,
            confidence=round(ext.confidence, 4),
            bbox=list(ext.detection.bbox),
            file=ext.filename,
        )
        for ext in extractions
    ]

    return ImageMetadata(
        source_image=source_path.name,
        processed_at=datetime.now().isoformat(),
        description=generate_description(labels),
        image_size={"height": image_size[0], "width": image_size[1]},
        num_objects=len(extractions),
        labels=sorted(set(labels)),
        objects=objects,
        ai_description=ai_description,
    )


def save_metadata(
    metadata: ImageMetadata,
    output_path: Path,
    indent: int = 2,
) -> None:
    """Save metadata to JSON file.

    Parameters
    ----------
    metadata : ImageMetadata
        Metadata to save.
    output_path : Path
        Output JSON file path.
    indent : int
        JSON indentation level.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = asdict(metadata)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_metadata(path: Path) -> ImageMetadata:
    """Load metadata from JSON file.

    Parameters
    ----------
    path : Path
        Path to JSON metadata file.

    Returns
    -------
    ImageMetadata
        Loaded metadata.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    objects = [ObjectMetadata(**obj) for obj in data.get("objects", [])]

    return ImageMetadata(
        source_image=data["source_image"],
        processed_at=data["processed_at"],
        description=data["description"],
        image_size=data["image_size"],
        num_objects=data["num_objects"],
        labels=data["labels"],
        objects=objects,
    )
