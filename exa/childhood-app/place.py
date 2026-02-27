#!/usr/bin/env python3
"""
Place CLI
=========

Place segments onto a canvas system for installation display.

Usage
-----
    # Edit config/place_config.yaml, then run:
    python place.py

    # Use custom config:
    python place.py --config my_config.yaml

    # Verbose output:
    python place.py --verbose

API Usage
---------
    from services.service_canvas import Canvas
    from models.model_placed_item import PlacedItem

    canvas = Canvas(8000, 3000)
    canvas.place("segment.png", 100, 200)  # x, y from bottom-left
    canvas.place_random("output/", count=50)
    canvas.save("final.png")
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from services.service_canvas import Canvas
from services.service_image_index import ImageIndex
from utils.util_load_config import load_config, get_nested

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = Path(__file__).parent / "config" / "place_config.yaml"


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Place segments onto canvas. Edit config/place_config.yaml to configure.",
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
    width = get_nested(cfg, "canvas", "width", default=8000)
    height = get_nested(cfg, "canvas", "height", default=3000)
    output = Path(get_nested(cfg, "output", "path", default="canvas.png"))
    background = get_nested(cfg, "canvas", "background", default="transparent")
    save_layout = get_nested(cfg, "output", "save_layout")

    # Random placement config
    random_enabled = get_nested(cfg, "random", "enabled", default=False)
    random_count = (
        get_nested(cfg, "random", "count", default=30)
        if random_enabled
        else None
    )
    from_dir = Path(get_nested(cfg, "random", "from_dir", default="./output"))
    obj_type = get_nested(cfg, "random", "object_type")

    # Cutout placement config
    cutout_enabled = get_nested(cfg, "place_cutout", "enabled", default=False)
    place_cutout = None
    if cutout_enabled:
        cutout_path = get_nested(cfg, "place_cutout", "metadata_path")
        if cutout_path:
            place_cutout = Path(cutout_path)
    cutout_index = get_nested(cfg, "place_cutout", "cutout_index", default=0)
    pos = get_nested(cfg, "place_cutout", "position", default=[0, 0])

    # Animation config
    delay = get_nested(cfg, "animation", "delay", default=1.0)
    jitter = get_nested(cfg, "animation", "jitter", default=0)
    order = get_nested(cfg, "animation", "order", default="sequential")
    animate = get_nested(cfg, "animation", "enabled", default=False)
    save_each = get_nested(cfg, "animation", "save_each")

    # Grid config
    grid_enabled = get_nested(cfg, "grid", "enabled", default=False)
    grid = get_nested(cfg, "grid", "columns") if grid_enabled else None

    # Layout config
    layout_path = get_nested(cfg, "layout", "load")
    layout = Path(layout_path) if layout_path else None

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # Parse background color
    if background == "transparent":
        bg = (0, 0, 0, 0)
    elif background == "white":
        bg = (255, 255, 255, 255)
    elif background == "black":
        bg = (0, 0, 0, 255)
    elif background.startswith("#"):
        hex_color = background.lstrip("#")
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        bg = (r, g, b, 255)
    else:
        bg = (0, 0, 0, 0)

    canvas = Canvas(width, height, bg)

    # Place cutout segments iteratively
    if place_cutout:
        metadata_path = place_cutout
        if not metadata_path.exists():
            logger.error("Metadata file not found: %s", metadata_path)
            return 1

        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)

        cutouts = metadata.get("cutouts", [])
        if not cutouts:
            logger.error("No cutouts found in metadata")
            return 1

        if cutout_index >= len(cutouts):
            logger.error(
                "Cutout index %d out of range (0-%d)",
                cutout_index,
                len(cutouts) - 1,
            )
            return 1

        cutout_data = cutouts[cutout_index]
        base_dir = metadata_path.parent

        x, y = pos
        logger.info(
            "Placing cutout '%s' at (%d, %d)",
            cutout_data.get("label", "unknown"),
            x,
            y,
        )
        if jitter > 0:
            logger.info("  Jitter: ±%dpx", jitter)
        if order != "sequential":
            logger.info("  Order: %s", order)
        if animate:
            logger.info("  Animation: enabled")

        canvas.place_cutout_segments(
            cutout_data,
            canvas_x=x,
            canvas_y=y,
            base_dir=base_dir,
            delay=delay,
            save_each=save_each,
            jitter=jitter,
            order=order,
            animate=animate,
        )

    # Load from layout file
    elif layout:
        canvas.load_layout(layout)

    # Random placement
    elif random_count:
        canvas.place_random(
            from_dir,
            random_count,
            obj_type,
            animate=animate,
            delay=delay,
            jitter=jitter,
        )

    # Grid layout
    elif grid:
        index = ImageIndex(from_dir)

        all_segs = []
        for entry in index.all():
            for cutout in entry.cutouts:
                if (
                    obj_type is None
                    or obj_type.lower() in cutout.label.lower()
                ):
                    all_segs.extend(cutout.segments)

        canvas.place_grid(all_segs, cols=grid)

    else:
        logger.error(
            "No placement mode enabled. Edit %s to enable random, place_cutout, grid, or layout.",
            args.config,
        )
        return 1

    # Save outputs
    if save_layout:
        canvas.save_layout(Path(save_layout))

    canvas.save(output)
    print(f"Saved {len(canvas)} items to {output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
