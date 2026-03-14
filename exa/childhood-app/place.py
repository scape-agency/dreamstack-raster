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
import random
import sys
from pathlib import Path

from services.service_canvas import Canvas
from services.service_image_index import ImageIndex
from utils.util_load_config import load_config, get_nested

# simple color‑filter helper used in place_random / place_grid

def _image_is_color(
    path: str | Path, color: str, threshold: int = 50
) -> bool:
    """Return True if the average color of *path* matches named color.

    The implementation is intentionally simple: convert to RGB and
    compare channel means.  "red" means R is greater than G and B by
    *threshold* pixels.  You can extend this with proper HSV ranges
    if you like.
    """
    from PIL import Image
    import numpy as np

    try:
        img = Image.open(path).convert("RGB")
    except Exception:  # file could be missing or unreadable
        return False

    arr = np.array(img)
    # arr shape (h,w,3)
    mean_r = float(arr[:, :, 0].mean())
    mean_g = float(arr[:, :, 1].mean())
    mean_b = float(arr[:, :, 2].mean())
    img.close()

    color = color.lower()
    if color == "red":
        return mean_r > mean_g + threshold and mean_r > mean_b + threshold
    if color == "green":
        return mean_g > mean_r + threshold and mean_g > mean_b + threshold
    if color == "blue":
        return mean_b > mean_r + threshold and mean_b > mean_g + threshold

    # unknown color, always allow
    return True


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
    keep_cutouts = get_nested(cfg, "random", "keep_cutouts", default=False)
    margin = get_nested(cfg, "random", "margin", default=50)

    # optional color filter (only used in random/grid modes)
    color_filter = get_nested(cfg, "filter", "color")
    color_threshold = get_nested(cfg, "filter", "color_threshold", default=50)

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
    # organic placement: when true, apply a small default per-segment jitter
    # so segments look slightly moved from their original positions
    organic = get_nested(cfg, "animation", "organic", default=True)
    organic_jitter = get_nested(cfg, "animation", "organic_jitter", default=6)
    # Fluid grid placement options
    selection_ratio = get_nested(cfg, "animation", "selection_ratio", default=0.7)
    rotation_jitter = get_nested(cfg, "animation", "rotation_jitter", default=3.0)

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
            organic=organic,
            organic_jitter=organic_jitter,
            order=order,
            animate=animate,
            selection_ratio=selection_ratio,
            rotation_jitter=rotation_jitter,
        )

    # Load from layout file
    elif layout:
        canvas.load_layout(layout)

    # Random placement
    elif random_count:
        if keep_cutouts:
            logger.info("Random placement: grouping by cutout (keep_cutouts=True)")
            # select random cutouts instead of individual segments
            index = ImageIndex(from_dir)
            index.load()

            candidates: list[dict] = []
            for entry in index.all():
                # we need the raw metadata to get positions/sizes
                with open(entry.metadata_path, encoding="utf-8") as f:
                    meta = json.load(f)
                for c in meta.get("cutouts", []):
                    label = c.get("label", "")
                    if obj_type is None or obj_type.lower() in label.lower():
                        candidates.append({
                            "cutout": c,
                            "base_dir": entry.metadata_path.parent,
                        })

            # apply color filtering to cutouts if requested
            if color_filter:
                filtered: list[dict] = []
                for item in candidates:
                    segs = item["cutout"].get("segments", [])
                    if any(_image_is_color(item["base_dir"] / s.get("file", ""), color_filter, color_threshold) for s in segs):
                        filtered.append(item)
                candidates = filtered

            if not candidates:
                logger.warning("No cutouts available for random placement")
            else:
                logger.info("Selected %d cutouts for placement", min(random_count, len(candidates)))
                chosen = random.sample(candidates, min(random_count, len(candidates)))
                for sel in chosen:
                    cutout_meta = sel["cutout"]
                    cutout_size = tuple(cutout_meta.get("size", [0, 0]))
                    # compute random canvas position ensuring whole cutout fits
                    x = random.randint(
                        margin,
                        max(margin, width - cutout_size[0] - margin),
                    )
                    y = random.randint(
                        margin,
                        max(margin, height - cutout_size[1] - margin),
                    )
                    canvas.place_cutout_segments(
                        cutout_meta,
                        canvas_x=x,
                        canvas_y=y,
                        base_dir=sel["base_dir"],
                        delay=delay,
                        save_each=save_each,
                        jitter=jitter,
                        organic=organic,
                        organic_jitter=organic_jitter,
                        order=order,
                        animate=animate,
                        selection_ratio=selection_ratio,
                        rotation_jitter=rotation_jitter,
                    )
        else:
            canvas.place_random(
                from_dir,
                random_count,
                obj_type,
                animate=animate,
                delay=delay,
                jitter=jitter,
                color=color_filter,
                color_threshold=color_threshold,
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
        # apply color filter if requested
        if color_filter:
            all_segs = [p for p in all_segs if _image_is_color(p, color_filter, color_threshold)]
            if not all_segs:
                logger.warning("No segments left after color filter")

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
