"""
Image Index Service
===================

Index of processed images for searching.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from models.model_cutout_result import CutoutResult
from models.model_search_result import SearchResult

logger = logging.getLogger(__name__)


class ImageIndex:
    """Index of processed images for searching.

    Loads all metadata.json files from output directory
    and provides search functionality.
    """

    def __init__(self, output_dir: str | Path) -> None:
        """Initialize index.

        Parameters
        ----------
        output_dir : str | Path
            Output directory containing processed images.
        """
        self.output_dir = Path(output_dir)
        self.entries: list[SearchResult] = []
        self.loaded = False

    def load(self) -> None:
        """Load all metadata files into index."""
        if self.loaded:
            return

        self.entries = []

        # Find all metadata.json files
        for metadata_path in self.output_dir.rglob("metadata.json"):
            try:
                entry = self.load_metadata(metadata_path)
                if entry:
                    self.entries.append(entry)
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.warning("Failed to load %s: %s", metadata_path, e)

        self.loaded = True
        logger.info("Loaded %d images into index", len(self.entries))

    def load_metadata(self, metadata_path: Path) -> SearchResult | None:
        """Load a single metadata file."""
        with open(metadata_path, encoding="utf-8") as f:
            data = json.load(f)

        output_dir = metadata_path.parent

        # Build cutout results
        cutouts = []
        for cutout_data in data.get("cutouts", []):
            cutout_path = output_dir / cutout_data["file"]
            segments = [
                output_dir / seg["file"]
                for seg in cutout_data.get("segments", [])
            ]

            cutouts.append(
                CutoutResult(
                    path=cutout_path,
                    label=cutout_data["label"],
                    confidence=cutout_data["confidence"],
                    segments=segments,
                    source_image=data["source_image"],
                    metadata_path=metadata_path,
                )
            )

        return SearchResult(
            source_image=data["source_image"],
            output_dir=output_dir,
            metadata_path=metadata_path,
            ai_description=data.get("ai_description", ""),
            detected_objects=data.get("detected_objects", []),
            cutouts=cutouts,
        )

    def all(self) -> list[SearchResult]:
        """Get all indexed images."""
        self.load()
        return self.entries.copy()

    def search_description(
        self,
        query: str,
        limit: int | None = None,
    ) -> list[SearchResult]:
        """Search images by AI description.

        Uses simple keyword matching. For better results,
        consider using embedding-based semantic search.

        Parameters
        ----------
        query : str
            Search query.
        limit : int | None
            Maximum results to return.

        Returns
        -------
        list[SearchResult]
            Matching results sorted by relevance.
        """
        self.load()

        query_words = set(query.lower().split())
        results = []

        for entry in self.entries:
            # Score based on keyword overlap
            desc_words = set(entry.ai_description.lower().split())
            obj_words = set(w.lower() for w in entry.detected_objects)

            # Count matches
            desc_matches = len(query_words & desc_words)
            obj_matches = len(query_words & obj_words)

            # Weight object matches higher
            score = desc_matches + (obj_matches * 2)

            if score > 0:
                entry.score = score
                results.append(entry)

        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)

        if limit:
            results = results[:limit]

        return results

    def search_type(
        self,
        object_type: str,
        limit: int | None = None,
    ) -> list[CutoutResult]:
        """Search cutouts by object type/label.

        Parameters
        ----------
        object_type : str
            Object type to search for (e.g., "person", "face").
        limit : int | None
            Maximum results to return.

        Returns
        -------
        list[CutoutResult]
            Matching cutouts sorted by confidence.
        """
        self.load()

        object_type = object_type.lower()
        results = []

        for entry in self.entries:
            for cutout in entry.cutouts:
                if object_type in cutout.label.lower():
                    results.append(cutout)

        # Sort by confidence
        results.sort(key=lambda x: x.confidence, reverse=True)

        if limit:
            results = results[:limit]

        return results

    def get_random_cutout(
        self,
        object_type: str | None = None,
    ) -> CutoutResult | None:
        """Get a random cutout, optionally filtered by type.

        Parameters
        ----------
        object_type : str | None
            Optional object type filter.

        Returns
        -------
        CutoutResult | None
            Random cutout or None if none available.
        """
        self.load()

        candidates = []
        for entry in self.entries:
            for cutout in entry.cutouts:
                if (
                    object_type is None
                    or object_type.lower() in cutout.label.lower()
                ):
                    candidates.append(cutout)

        if not candidates:
            return None

        return random.choice(candidates)

    def get_random_segment(
        self,
        object_type: str | None = None,
    ) -> Path | None:
        """Get a random segment image path.

        Parameters
        ----------
        object_type : str | None
            Optional object type filter.

        Returns
        -------
        Path | None
            Path to random segment image or None.
        """
        cutout = self.get_random_cutout(object_type)
        if cutout is None or not cutout.segments:
            return None

        return random.choice(cutout.segments)

    def stats(self) -> dict:
        """Get index statistics."""
        self.load()

        total_cutouts = sum(len(e.cutouts) for e in self.entries)
        total_segments = sum(
            len(c.segments) for e in self.entries for c in e.cutouts
        )

        # Count object types
        type_counts: dict[str, int] = {}
        for entry in self.entries:
            for cutout in entry.cutouts:
                label = cutout.label.lower()
                type_counts[label] = type_counts.get(label, 0) + 1

        return {
            "total_images": len(self.entries),
            "total_cutouts": total_cutouts,
            "total_segments": total_segments,
            "object_types": type_counts,
        }
