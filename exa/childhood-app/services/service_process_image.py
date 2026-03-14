"""
Process Image Service
=====================

Process a single image through the full preprocessing pipeline.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

from models.model_app_config import AppConfig
from services.service_describe_image import describe_image
from services.service_detect_objects import detect_objects
from services.service_extract_cutout import extract_cutout
from services.service_segment_image import segment_image
from services.service_apply_effects import apply_effects

logger = logging.getLogger(__name__)


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

    Parameters
    ----------
    image_path : Path
        Path to the input image.
    output_dir : Path
        Output directory for results.
    input_dir : Path
        Input directory (used to preserve folder structure).
    config : AppConfig
        Application configuration.
    apply_effects_flag : bool
        Whether to apply effects to segments. Default True.
    copy_source : bool
        Whether to copy source image to output. Default True.

    Returns
    -------
    dict[str, Any]
        Metadata dictionary containing processing results.
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
