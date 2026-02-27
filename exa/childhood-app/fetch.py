#!/usr/bin/env python3
"""
Fetch CLI
=========

Query processed images by description or object type.

Usage
-----
    # Edit config/fetch_config.yaml, then run:
    python fetch.py

    # Use custom config:
    python fetch.py --config my_config.yaml

API Usage
---------
    from services.service_image_index import ImageIndex
    from utils.util_fetch_by_description import fetch_by_description
    from utils.util_fetch_by_type import fetch_by_type

    index = ImageIndex("./output")
    results = index.search_description("child playing")
    for result in results:
        print(result.source_image, result.score)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from services.service_image_index import ImageIndex
from utils.util_load_config import load_config, get_nested

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = Path(__file__).parent / "config" / "fetch_config.yaml"


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Query processed images. Edit config/fetch_config.yaml to configure.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Config file (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    if not cfg:
        logger.error("Config file not found: %s", args.config)
        return 1

    # Read all settings from config
    output_dir = Path(get_nested(cfg, "output_dir", default="./output"))
    query = get_nested(cfg, "search", "query")
    obj_type = get_nested(cfg, "search", "type")
    limit = get_nested(cfg, "options", "limit", default=10)
    cutout_mode = get_nested(cfg, "options", "cutout", default=False)
    json_output = get_nested(cfg, "options", "json_output", default=False)
    random_mode = get_nested(cfg, "search", "random", default=False)
    list_mode = get_nested(cfg, "search", "list", default=False)
    stats_mode = get_nested(cfg, "search", "stats", default=False)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    index = ImageIndex(output_dir)

    # Stats mode
    if stats_mode:
        stats = index.stats()
        if json_output:
            print(json.dumps(stats, indent=2))
        else:
            print(f"Total images: {stats['total_images']}")
            print(f"Total cutouts: {stats['total_cutouts']}")
            print(f"Total segments: {stats['total_segments']}")
            print("\nObject types:")
            for obj_t, count in sorted(
                stats["object_types"].items(),
                key=lambda x: x[1],
                reverse=True,
            ):
                print(f"  {obj_t}: {count}")
        return 0

    # List mode
    if list_mode:
        entries = index.all()
        if json_output:
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
    if random_mode:
        if obj_type:
            cutout = index.get_random_cutout(obj_type)
            if cutout:
                if json_output:
                    print(json.dumps(cutout.to_dict(), indent=2))
                else:
                    print(f"Path: {cutout.path}")
                    print(f"Label: {cutout.label}")
                    print(f"Segments: {len(cutout.segments)}")
            else:
                print("No matching cutouts found")
                return 1
        else:
            segment = index.get_random_segment(obj_type)
            if segment:
                print(segment)
            else:
                print("No segments found")
                return 1
        return 0

    # Search by type
    if obj_type:
        results = index.search_type(obj_type, limit)
        if json_output:
            print(json.dumps([r.to_dict() for r in results], indent=2))
        else:
            print(f"Found {len(results)} cutouts for '{obj_type}':")
            for r in results:
                print(f"  {r.path} (conf: {r.confidence:.2f})")
        return 0

    # Search by description
    if query:
        results = index.search_description(query, limit)

        if cutout_mode:
            # Flatten to cutouts
            cutouts = [c for r in results for c in r.cutouts]
            if json_output:
                print(json.dumps([c.to_dict() for c in cutouts], indent=2))
            else:
                print(f"Found {len(cutouts)} cutouts matching '{query}':")
                for c in cutouts:
                    print(f"  {c.path}")
        else:
            if json_output:
                print(json.dumps([r.to_dict() for r in results], indent=2))
            else:
                print(f"Found {len(results)} images matching '{query}':")
                for r in results:
                    print(f"  {r.source_image} (score: {r.score})")
                    print(f"    {r.ai_description[:80]}...")
        return 0

    # No action specified
    logger.error("No search mode enabled. Edit %s to set query, type, list, stats, or random.", args.config)
    return 1


if __name__ == "__main__":
    sys.exit(main())
