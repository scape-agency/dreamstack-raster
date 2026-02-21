#!/usr/bin/env python3
"""
Childhood App Preprocessor
==========================

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
    python preprocess.py                          # Default: ./images -> ./output
    python preprocess.py --input ./photos         # Custom input
    python preprocess.py --no-effects             # Skip effects
    python preprocess.py --vision-backend openai  # Use OpenAI for description

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


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

# Load environment variables
try:
    # pylint: disable=import-outside-toplevel
    from dotenv import load_dotenv  # type: ignore[assignment]
except ImportError:
    # dotenv is optional
    def load_dotenv(*_args: object, **_kwargs: object) -> bool:  # type: ignore[misc]
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
# pylint: disable=wrong-import-position
from modules.config import AppConfig, CutoutConfig, EffectConfig, SegmentConfig
from modules.effects import apply_effects
from modules.grid import segment_image

# pylint: enable=wrong-import-position

logger = logging.getLogger(__name__)

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def find_images(directory: Path, recursive: bool = True) -> list[Path]:
    """Find all image files in directory."""
    pattern = "**/*" if recursive else "*"
    images = []
    for path in directory.glob(pattern):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)
    return sorted(images)


def describe_image(
    image_path: Path,
    backend: str = "openai",
) -> tuple[str, list[str]]:
    """Describe image using AI vision.

    Returns (description, object_list).
    """
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.detection.describer import (
        DescriptionConfig,
        ImageDescriber,
    )

    config = DescriptionConfig(backend=backend)  # type: ignore[arg-type]
    describer = ImageDescriber(config)

    result = describer.describe(image_path)
    return result.description, result.objects


def detect_objects(
    image: np.ndarray,
    backend: str = "ultralytics",
    confidence: float = 0.5,
    prompts: list[str] | None = None,
) -> list[dict]:
    """Detect objects in image.

    Returns list of detections with bbox and label.
    """
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.detection import DetectionConfig, create_detector

    config = DetectionConfig(
        backend=backend,  # type: ignore[arg-type]
        confidence_threshold=confidence,
        text_prompts=prompts,
    )

    detector = create_detector(config)
    result = detector.detect(image)

    detections = []
    for det in result.detections:
        detections.append(
            {
                "label": det.label,
                "confidence": det.confidence,
                "bbox": det.bbox,  # (x, y, w, h)
            }
        )

    return detections


def extract_cutout(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    margin: int = 50,
    max_size: int = 1200,
    segment_size: tuple[int, int] = (250, 250),
    segment_align: bool = True,
) -> tuple[Image.Image, dict]:
    """Extract and scale bounding box cutout with smart sizing.

    The largest dimension is scaled to max_size.
    The smallest dimension is rounded up to a multiple of segment size.
    Bounding box is expanded equally on both sides, filling with
    transparent pixels where the source image has no data.

    Parameters
    ----------
    image : np.ndarray
        Source image (BGR).
    bbox : tuple[int, int, int, int]
        Bounding box (x, y, width, height).
    margin : int
        Margin around bbox (also max random offset). Default 50.
    max_size : int
        Maximum size for largest dimension. Default 1200.
    segment_size : tuple[int, int]
        Segment size for alignment. Default (250, 250).
    segment_align : bool
        Align smallest dimension to segment multiple. Default True.

    Returns
    -------
    tuple[Image.Image, dict]
        Scaled cutout with alpha channel and metadata about sizing.
    """
    img_h, img_w = image.shape[:2]
    x, y, bw, bh = bbox

    # Add margin to bounding box
    x1_desired = x - margin
    y1_desired = y - margin
    x2_desired = x + bw + margin
    y2_desired = y + bh + margin

    crop_w = x2_desired - x1_desired
    crop_h = y2_desired - y1_desired

    # Determine target dimensions
    # Largest dimension -> max_size
    if crop_w >= crop_h:
        # Width is largest
        scale = max_size / crop_w
        target_w = max_size
        target_h = int(crop_h * scale)
    else:
        # Height is largest
        scale = max_size / crop_h
        target_h = max_size
        target_w = int(crop_w * scale)

    # Align smallest dimension to segment multiple
    if segment_align:
        seg_w, seg_h = segment_size
        if target_w <= target_h:
            # Width is smallest, align to seg_w
            target_w = ((target_w + seg_w - 1) // seg_w) * seg_w
        else:
            # Height is smallest, align to seg_h
            target_h = ((target_h + seg_h - 1) // seg_h) * seg_h

    # Recalculate crop dimensions to match target aspect ratio
    target_aspect = target_w / target_h
    crop_aspect = crop_w / crop_h

    if target_aspect > crop_aspect:
        # Need wider crop
        new_crop_w = int(crop_h * target_aspect)
        expand = (new_crop_w - crop_w) // 2
        x1_desired -= expand
        x2_desired += expand
        crop_w = new_crop_w
    elif target_aspect < crop_aspect:
        # Need taller crop
        new_crop_h = int(crop_w / target_aspect)
        expand = (new_crop_h - crop_h) // 2
        y1_desired -= expand
        y2_desired += expand
        crop_h = new_crop_h

    # Calculate actual crop bounds (clamped to image)
    x1_actual = max(0, x1_desired)
    y1_actual = max(0, y1_desired)
    x2_actual = min(img_w, x2_desired)
    y2_actual = min(img_h, y2_desired)

    # Crop from source image
    crop = image[y1_actual:y2_actual, x1_actual:x2_actual]

    # Convert BGR to RGB
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    # Convert to PIL with alpha
    pil_crop = Image.fromarray(crop_rgb).convert("RGBA")

    # Create full-size canvas with transparency
    full_crop = Image.new("RGBA", (crop_w, crop_h), (0, 0, 0, 0))

    # Calculate paste position for the actual crop
    paste_x = x1_actual - x1_desired
    paste_y = y1_actual - y1_desired
    full_crop.paste(pil_crop, (paste_x, paste_y))

    # Scale to target size
    result = full_crop.resize((target_w, target_h), Image.Resampling.LANCZOS)

    # Metadata about the cutout
    metadata = {
        "original_bbox": [x, y, bw, bh],
        "margin": margin,
        "crop_bounds": [x1_desired, y1_desired, x2_desired, y2_desired],
        "actual_bounds": [x1_actual, y1_actual, x2_actual, y2_actual],
        "target_size": [target_w, target_h],
        "has_padding": (
            x1_desired < 0
            or y1_desired < 0
            or x2_desired > img_w
            or y2_desired > img_h
        ),
    }

    return result, metadata


def process_image(
    image_path: Path,
    output_dir: Path,
    input_dir: Path,
    config: AppConfig,
    *,
    apply_effects_flag: bool = True,
    copy_source: bool = True,
) -> dict[str, Any]:
    """Process a single image through the pipeline.

    Returns metadata dict.
    """
    logger.info("Processing: %s", image_path.name)

    # Preserve folder structure
    try:
        relative_parent = image_path.parent.relative_to(input_dir)
        image_output_dir = output_dir / relative_parent / image_path.stem
    except ValueError:
        image_output_dir = output_dir / image_path.stem

    image_output_dir.mkdir(parents=True, exist_ok=True)
    cutouts_dir = image_output_dir / "cutouts"
    segments_dir = image_output_dir / "segments"
    cutouts_dir.mkdir(exist_ok=True)
    segments_dir.mkdir(exist_ok=True)

    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    h, w = image.shape[:2]

    # Step 1: Describe image with AI
    logger.info("  Describing with AI...")
    try:
        description, objects = describe_image(
            image_path,
            backend=config.vision_backend,
        )
        logger.info("  Found objects: %s...", ", ".join(objects[:5]))
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning("  AI description failed: %s", e)
        description = ""
        objects = []

    # Step 2: Detect objects
    logger.info("  Detecting objects...")
    detections = detect_objects(
        image,
        backend=config.detection_backend,
        confidence=config.confidence_threshold,
        prompts=objects if objects else None,
    )
    logger.info("  Detected %d objects", len(detections))

    # Process each detection
    cutout_metadata = []

    for i, det in enumerate(detections):
        label = det["label"]
        bbox = det["bbox"]
        confidence = det["confidence"]

        cutout_name = f"{label}_{i + 1}"
        cutout_filename = f"{cutout_name}.png"

        logger.info("  Processing cutout: %s", cutout_name)

        # Step 3: Extract and scale cutout with smart sizing
        cutout, cutout_info = extract_cutout(
            image,
            bbox,
            margin=config.cutout.margin,
            max_size=config.cutout.max_size,
            segment_size=config.segment.segment_size,
            segment_align=config.cutout.segment_align,
        )

        logger.info("    Cutout size: %dx%d", cutout.width, cutout.height)

        # Save cutout
        cutout_path = cutouts_dir / cutout_filename
        cutout.save(cutout_path)

        # Step 4: Segment cutout
        logger.info("    Segmenting into grid...")
        segments = segment_image(cutout, config.segment)

        segment_output_dir = segments_dir / cutout_name
        segment_output_dir.mkdir(exist_ok=True)

        segment_metadata = []

        for seg in segments:
            # Step 5: Apply effects
            if apply_effects_flag:
                effect_result = apply_effects(seg.image, config.effects)
                seg_image = effect_result.image
                effects_applied = effect_result.effects_applied
            else:
                seg_image = seg.image
                effects_applied = []

            # Save segment
            seg_path = segment_output_dir / seg.filename
            seg_image.save(seg_path)

            segment_metadata.append(
                {
                    "file": f"segments/{cutout_name}/{seg.filename}",
                    **seg.to_dict(),
                    "effects": effects_applied,
                }
            )

        cutout_metadata.append(
            {
                "label": label,
                "confidence": round(confidence, 4),
                "file": f"cutouts/{cutout_filename}",
                "size": [cutout.width, cutout.height],
                **cutout_info,
                "segments": segment_metadata,
            }
        )

    # Copy source if requested
    if copy_source:
        shutil.copy2(
            image_path, image_output_dir / f"source{image_path.suffix}"
        )

    # Build metadata
    metadata = {
        "source_image": image_path.name,
        "processed_at": datetime.now().isoformat(),
        "image_size": {"width": w, "height": h},
        "ai_description": description,
        "detected_objects": objects,
        "config": config.to_dict(),
        "cutouts": cutout_metadata,
    }

    # Save metadata
    metadata_path = image_output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Preprocess images for the childhood art installation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path(__file__).parent / "images",
        help="Input directory (default: ./images)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path(__file__).parent / "output",
        help="Output directory (default: ./output)",
    )
    parser.add_argument(
        "--vision-backend",
        type=str,
        default="openai",
        choices=["openai", "mistral"],
        help="AI vision backend (default: openai)",
    )
    parser.add_argument(
        "--detection-backend",
        type=str,
        default="ultralytics",
        choices=["ultralytics", "grounding_dino_sam"],
        help="Detection backend (default: ultralytics)",
    )
    parser.add_argument(
        "--confidence",
        "-c",
        type=float,
        default=0.5,
        help="Detection confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--cutout-size",
        type=int,
        default=1200,
        help="Maximum cutout size in pixels (default: 1200)",
    )
    parser.add_argument(
        "--segment-size",
        type=str,
        default="400x300",
        help="Segment size WxH (default: 400x300)",
    )
    parser.add_argument(
        "--inbetweens",
        action="store_true",
        help="Generate horizontal and vertical in-between segments (~2x more)",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=50,
        help="Margin around cutouts (default: 50)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of images to process",
    )
    parser.add_argument(
        "--no-effects",
        action="store_true",
        help="Skip applying effects to segments",
    )
    parser.add_argument(
        "--no-shadow",
        action="store_true",
        help="Skip drop shadow effect",
    )
    parser.add_argument(
        "--filters",
        type=str,
        nargs="*",
        default=[],
        help="Filters to apply (warm_filter, cool_filter, vintage, etc.)",
    )
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        default=True,
        help="Process subdirectories recursively (default: True)",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_false",
        dest="recursive",
        help="Do not process subdirectories",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Validate input
    if not args.input.exists():
        logger.error("Input directory not found: %s", args.input)
        return 1

    # Parse segment size (WxH format)
    try:
        seg_parts = args.segment_size.lower().split("x")
        if len(seg_parts) == 2:
            seg_w, seg_h = int(seg_parts[0]), int(seg_parts[1])
        else:
            seg_w = seg_h = int(seg_parts[0])
    except ValueError:
        logger.error("Invalid segment size format: %s", args.segment_size)
        return 1

    # Build configuration
    config = AppConfig(
        cutout=CutoutConfig(
            max_size=args.cutout_size,
            margin=args.margin,
        ),
        segment=SegmentConfig(
            segment_size=(seg_w, seg_h),
            generate_inbetweens=args.inbetweens,
        ),
        effects=EffectConfig(
            drop_shadow=not args.no_shadow,
            filters=args.filters,
        ),
        vision_backend=args.vision_backend,
        detection_backend=args.detection_backend,
        confidence_threshold=args.confidence,
    )

    logger.info("Childhood App Preprocessor")
    logger.info("=" * 50)
    logger.info("Input: %s", args.input)
    logger.info("Output: %s", args.output)
    logger.info("Vision: %s", config.vision_backend)
    logger.info("Detection: %s", config.detection_backend)
    logger.info("Cutout max size: %s", config.cutout.max_size)
    logger.info("Segment size: %s", config.segment.segment_size)
    if config.segment.generate_inbetweens:
        logger.info("In-between segments: ENABLED (H+V)")
    logger.info("=" * 50)

    # Find images
    images = find_images(args.input, args.recursive)
    total = len(images)

    if total == 0:
        logger.warning("No images found in %s", args.input)
        return 0

    # Apply limit
    if args.limit and args.limit < total:
        images = images[: args.limit]
        logger.info("Limited to %d of %d images", args.limit, total)
        total = args.limit

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
                args.output,
                args.input,
                config,
                apply_effects_flag=not args.no_effects,
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
    print(f"Output saved to: {args.output}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
