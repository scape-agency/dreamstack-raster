#!/usr/bin/env python3
"""
Fetch Module
============

Query processed images by description or object type.

Usage
-----
    # Search by description
    python fetch.py --query "child playing outside"

    # Search by object type
    python fetch.py --type person --limit 5

    # Get best matching cutout
    python fetch.py --query "smiling face" --cutout

    # List all processed images
    python fetch.py --list

API Usage
---------
    from fetch import ImageIndex, fetch_by_description, fetch_by_type

    index = ImageIndex("./output")
    results = index.search_description("child playing")
    for result in results:
        print(result.source_image, result.score)
"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)


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
        self._entries: list[SearchResult] = []
        self._loaded = False

    def load(self) -> None:
        """Load all metadata files into index."""
        if self._loaded:
            return

        self._entries = []

        # Find all metadata.json files
        for metadata_path in self.output_dir.rglob("metadata.json"):
            try:
                entry = self._load_metadata(metadata_path)
                if entry:
                    self._entries.append(entry)
            except Exception as e:
                logger.warning(f"Failed to load {metadata_path}: {e}")

        self._loaded = True
        logger.info(f"Loaded {len(self._entries)} images into index")

    def _load_metadata(self, metadata_path: Path) -> SearchResult | None:
        """Load a single metadata file."""
        with open(metadata_path) as f:
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
        return self._entries.copy()

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

        for entry in self._entries:
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

        for entry in self._entries:
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
        import random

        self.load()

        candidates = []
        for entry in self._entries:
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
        import random

        cutout = self.get_random_cutout(object_type)
        if cutout is None or not cutout.segments:
            return None

        return random.choice(cutout.segments)

    def stats(self) -> dict:
        """Get index statistics."""
        self.load()

        total_cutouts = sum(len(e.cutouts) for e in self._entries)
        total_segments = sum(
            len(c.segments) for e in self._entries for c in e.cutouts
        )

        # Count object types
        type_counts: dict[str, int] = {}
        for entry in self._entries:
            for cutout in entry.cutouts:
                label = cutout.label.lower()
                type_counts[label] = type_counts.get(label, 0) + 1

        return {
            "total_images": len(self._entries),
            "total_cutouts": total_cutouts,
            "total_segments": total_segments,
            "object_types": type_counts,
        }


# Convenience functions
def fetch_by_description(
    query: str,
    output_dir: str | Path = "./output",
    limit: int | None = None,
) -> list[SearchResult]:
    """Search images by description.

    Parameters
    ----------
    query : str
        Search query.
    output_dir : str | Path
        Output directory to search.
    limit : int | None
        Maximum results.

    Returns
    -------
    list[SearchResult]
        Matching results.
    """
    index = ImageIndex(output_dir)
    return index.search_description(query, limit)


def fetch_by_type(
    object_type: str,
    output_dir: str | Path = "./output",
    limit: int | None = None,
) -> list[CutoutResult]:
    """Search cutouts by object type.

    Parameters
    ----------
    object_type : str
        Object type to search for.
    output_dir : str | Path
        Output directory to search.
    limit : int | None
        Maximum results.

    Returns
    -------
    list[CutoutResult]
        Matching cutouts.
    """
    index = ImageIndex(output_dir)
    return index.search_type(object_type, limit)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Query processed images by description or type",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path(__file__).parent / "output",
        help="Output directory to search (default: ./output)",
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        help="Search by description query",
    )
    parser.add_argument(
        "--type",
        "-t",
        type=str,
        help="Search by object type",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=10,
        help="Maximum results (default: 10)",
    )
    parser.add_argument(
        "--cutout",
        action="store_true",
        help="Return cutout paths instead of image results",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Get a random result",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all processed images",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show index statistics",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    index = ImageIndex(args.output)

    # Stats mode
    if args.stats:
        stats = index.stats()
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print(f"Total images: {stats['total_images']}")
            print(f"Total cutouts: {stats['total_cutouts']}")
            print(f"Total segments: {stats['total_segments']}")
            print("\nObject types:")
            for obj_type, count in sorted(
                stats["object_types"].items(),
                key=lambda x: x[1],
                reverse=True,
            ):
                print(f"  {obj_type}: {count}")
        return 0

    # List mode
    if args.list:
        entries = index.all()
        if args.json:
            print(json.dumps([e.to_dict() for e in entries], indent=2))
        else:
            for entry in entries:
                print(f"{entry.source_image}")
                print(f"  Dir: {entry.output_dir}")
                print(f"  Description: {entry.ai_description[:100]}...")
                print(f"  Cutouts: {len(entry.cutouts)}")
                print()
        return 0

    # Random mode
    if args.random:
        if args.type:
            cutout = index.get_random_cutout(args.type)
            if cutout:
                if args.json:
                    print(json.dumps(cutout.to_dict(), indent=2))
                else:
                    print(f"Path: {cutout.path}")
                    print(f"Label: {cutout.label}")
                    print(f"Segments: {len(cutout.segments)}")
            else:
                print("No matching cutouts found")
                return 1
        else:
            segment = index.get_random_segment(args.type)
            if segment:
                print(segment)
            else:
                print("No segments found")
                return 1
        return 0

    # Search by type
    if args.type:
        results = index.search_type(args.type, args.limit)
        if args.json:
            print(json.dumps([r.to_dict() for r in results], indent=2))
        else:
            print(f"Found {len(results)} cutouts for '{args.type}':")
            for r in results:
                print(f"  {r.path} (conf: {r.confidence:.2f})")
        return 0

    # Search by description
    if args.query:
        results = index.search_description(args.query, args.limit)

        if args.cutout:
            # Flatten to cutouts
            cutouts = [c for r in results for c in r.cutouts]
            if args.json:
                print(json.dumps([c.to_dict() for c in cutouts], indent=2))
            else:
                print(f"Found {len(cutouts)} cutouts matching '{args.query}':")
                for c in cutouts:
                    print(f"  {c.path}")
        else:
            if args.json:
                print(json.dumps([r.to_dict() for r in results], indent=2))
            else:
                print(f"Found {len(results)} images matching '{args.query}':")
                for r in results:
                    print(f"  {r.source_image} (score: {r.score})")
                    print(f"    {r.ai_description[:80]}...")
        return 0

    # No action specified
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
