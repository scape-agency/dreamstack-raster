#!/usr/bin/env python3
"""
Childhood App Preprocessor CLI
==============================

Image preprocessing pipeline for the art installation.

Pipeline Steps:
1. Describe image with AI vision
2. Detect objects (bounding box cutouts)
3. Scale cutouts to 1000x1000px
4. Segment cutouts into randomized grid (~250x250px)
5. Apply visual effects (filters, drop shadows)
6. Save all outputs with metadata

Usage
-----
    # Edit config/preprocess_config.yaml, then run:
    python preprocess.py

    # Use custom config:
    python preprocess.py --config my_config.yaml

Output Structure
----------------
    output/
    ├── {folder}/
    │   └── {image_stem}/
    │       ├── metadata.json
    │       ├── source.jpg              # Original image copy
    │       ├── cutouts/
    │       │   ├── person_1.png        # 1000x1000 cutout
    │       │   └── face_1.png
    │       └── segments/
    │           ├── person_1/
    │           │   ├── seg_0_0.png
    │           │   ├── seg_0_1.png
    │           │   └── ...
    │           └── face_1/
    │               └── ...
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv
except ImportError:

    def load_dotenv(*_args, **_kwargs) -> bool:
        """Fallback when python-dotenv is not installed."""
        return False


# Add paths
project_root = Path(__file__).parent.parent.parent
env_file = project_root / ".env"
if env_file.exists():
    load_dotenv(env_file)
else:
    load_dotenv()

src_path = project_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

# Local modules
sys.path.insert(0, str(Path(__file__).parent))

from models.model_app_config import AppConfig
from models.model_cutout_config import CutoutConfig
from models.model_segment_config import SegmentConfig
from models.model_effect_config import EffectConfig
from services.service_process_image import process_image
from utils.util_find_images import find_images
from utils.util_load_config import load_config, get_nested

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = Path(__file__).parent / "config" / "preprocess_config.yaml"


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Preprocess images. Edit config/preprocess_config.yaml to configure.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
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
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    if not cfg:
        logger.error("Config file not found: %s", args.config)
        return 1

    # Read all settings from config
    input_dir = Path(get_nested(cfg, "paths", "input", default="./images"))
    output_dir = Path(get_nested(cfg, "paths", "output", default="./output"))
    recursive = get_nested(cfg, "paths", "recursive", default=True)

    vision_backend = get_nested(cfg, "backends", "vision", default="openai")
    detection_backend = get_nested(cfg, "backends", "detection", default="ultralytics")
    confidence = get_nested(cfg, "backends", "confidence", default=0.5)

    cutout_size = get_nested(cfg, "cutout", "max_size", default=1200)
    margin = get_nested(cfg, "cutout", "margin", default=50)
    cutout_mode = get_nested(cfg, "cutout", "cutout_mode", default="bbox")

    segment_size_cfg = get_nested(cfg, "segment", "size", default=[400, 300])
    seg_w, seg_h = segment_size_cfg[0], segment_size_cfg[1]
    randomize_offset = get_nested(cfg, "segment", "randomize_offset", default=False)
    max_offset = get_nested(cfg, "segment", "max_offset", default=50)
    inbetweens = get_nested(cfg, "segment", "generate_inbetweens", default=False)
    diagonal_inbetweens = get_nested(cfg, "segment", "generate_diagonal_inbetweens", default=False)
    # Fluid grid options
    fluid_grid = get_nested(cfg, "segment", "fluid_grid", default=True)
    size_variation = get_nested(cfg, "segment", "size_variation", default=0.3)
    layer_count = get_nested(cfg, "segment", "layer_count", default=2)
    layer_selection_ratio = get_nested(cfg, "segment", "layer_selection_ratio", default=0.7)
    rotation_range = get_nested(cfg, "segment", "rotation_range", default=5.0)
    contour_padding = get_nested(cfg, "segment", "contour_padding", default=0.15)

    effects_enabled = get_nested(cfg, "effects", "enabled", default=True)
    drop_shadow = get_nested(cfg, "effects", "drop_shadow", default=True)
    filters = get_nested(cfg, "effects", "filters", default=[])

    limit = get_nested(cfg, "limits", "max_images")

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Validate input
    if not input_dir.exists():
        logger.error("Input directory not found: %s", input_dir)
        return 1

    # Wipe entire output directory to remove stale segments, metadata, and
    # artifacts from previous runs (including images beyond max_images limit).
    if output_dir.exists():
        shutil.rmtree(output_dir)
        logger.info("Cleaned previous output directory: %s", output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build configuration
    config = AppConfig(
        cutout=CutoutConfig(
            max_size=cutout_size,
            margin=margin,
            cutout_mode=cutout_mode,
        ),
        segment=SegmentConfig(
            segment_size=(seg_w, seg_h),
            randomize_offset=randomize_offset,
            max_offset=max_offset,
            generate_inbetweens=inbetweens,
            generate_diagonal_inbetweens=diagonal_inbetweens,
            fluid_grid=fluid_grid,
            size_variation=size_variation,
            layer_count=layer_count,
            layer_selection_ratio=layer_selection_ratio,
            rotation_range=rotation_range,
            contour_padding=contour_padding,
        ),
        effects=EffectConfig(
            drop_shadow=drop_shadow,
            filters=filters,
        ),
        vision_backend=vision_backend,
        detection_backend=detection_backend,
        confidence_threshold=confidence,
    )

    logger.info("Childhood App Preprocessor")
    logger.info("=" * 50)
    logger.info("Input: %s", input_dir)
    logger.info("Output: %s", output_dir)
    logger.info("Vision: %s", config.vision_backend)
    logger.info("Detection: %s", config.detection_backend)
    logger.info("Cutout max size: %s", config.cutout.max_size)
    logger.info("Cutout mode: %s", config.cutout.cutout_mode)
    logger.info("Segment size: %s", config.segment.segment_size)
    if config.segment.fluid_grid:
        logger.info("Fluid grid: ENABLED (variation=%.0f%%, layers=%d, selection=%.0f%%)",
                    config.segment.size_variation * 100,
                    config.segment.layer_count,
                    config.segment.layer_selection_ratio * 100)
        if config.segment.rotation_range > 0:
            logger.info("Rotation range: ±%.1f°", config.segment.rotation_range)
    elif config.segment.generate_inbetweens or config.segment.generate_diagonal_inbetweens:
        types = []
        if config.segment.generate_inbetweens:
            types.append("H+V")
        if config.segment.generate_diagonal_inbetweens:
            types.append("D")
        logger.info("In-between segments: ENABLED (%s)", "+".join(types))
    if config.segment.randomize_offset:
        logger.info("Randomize offset: ENABLED (max %dpx)", config.segment.max_offset)
    logger.info("=" * 50)

    # Find images
    images = find_images(input_dir, recursive)
    total = len(images)

    if total == 0:
        logger.warning("No images found in %s", input_dir)
        return 0

    # Apply limit
    if limit and limit < total:
        images = images[:limit]
        logger.info("Limited to %d of %d images", limit, total)
        total = limit

    logger.info("Processing %d images", total)

    # Process images
    successful = 0
    failed = 0
    total_cutouts = 0
    total_segments = 0

    for i, image_path in enumerate(images):
        print(f"[{i + 1}/{total}] {image_path.name}")

        try:
            metadata = process_image(
                image_path,
                output_dir,
                input_dir,
                config,
                apply_effects_flag=effects_enabled,
            )

            num_cutouts = len(metadata["cutouts"])
            num_segments = sum(len(c["segments"]) for c in metadata["cutouts"])

            total_cutouts += num_cutouts
            total_segments += num_segments
            successful += 1

            logger.info(
                "  Created %d cutouts, %d segments", num_cutouts, num_segments
            )

        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("  Failed: %s", e)
            failed += 1

    # Summary
    print()
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Images processed: {successful}/{total}")
    print(f"Total cutouts: {total_cutouts}")
    print(f"Total segments: {total_segments}")
    print(f"Failed: {failed}")
    print()
    print(f"Output saved to: {output_dir}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
