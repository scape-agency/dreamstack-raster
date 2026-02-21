#!/usr/bin/env python3
"""
Childhood App
=============

Object detection and extraction pipeline for processing
image folders and extracting detected objects.

Usage
-----
    python main.py                    # Process ./images -> ./output
    python main.py --input ./photos   # Custom input folder
    python main.py --confidence 0.6   # Higher confidence threshold
    python main.py --model yolov8s-seg  # Larger model

Output Structure
----------------
For each input image, creates a folder containing:
- metadata.json: Description and object list
- {label}_{n}.png: Extracted objects with transparent background

Example:
    output/
    ├── photo_001/
    │   ├── metadata.json
    │   ├── dog_1.png
    │   ├── person_1.png
    │   └── bicycle_1.png
    └── photo_002/
        ├── metadata.json
        └── car_1.png

Requirements
------------
pip install ultralytics

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Load environment variables from .env
try:
    from dotenv import load_dotenv  # type: ignore[assignment]
except ImportError:
    # dotenv is optional
    def load_dotenv(*_args: object, **_kwargs: object) -> bool:  # type: ignore[misc]
        """Fallback when python-dotenv is not installed."""
        return False


# Try loading from project root
project_root = Path(__file__).parent.parent.parent
env_file = project_root / ".env"
if env_file.exists():
    load_dotenv(env_file)
else:
    load_dotenv()  # Try current directory

# Add src to path for development
src_path = Path(__file__).parent.parent.parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Detect and extract objects from images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path(__file__).parent / "images",
        help="Input directory containing images (default: ./images)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path(__file__).parent / "output",
        help="Output directory for results (default: ./output)",
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="yolov8n-seg",
        help="YOLO model name (default: yolov8n-seg)",
    )

    parser.add_argument(
        "--confidence",
        "-c",
        type=float,
        default=0.5,
        help="Minimum detection confidence (default: 0.5)",
    )

    parser.add_argument(
        "--margin",
        type=int,
        default=10,
        help="Margin around extracted objects in pixels (default: 10)",
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
        "--copy-source",
        action="store_true",
        help="Copy source images to output folders",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device for inference (default: auto)",
    )

    parser.add_argument(
        "--backend",
        "-b",
        type=str,
        default="ultralytics",
        choices=["ultralytics", "grounding_dino_sam"],
        help="Detection backend (default: ultralytics)",
    )

    parser.add_argument(
        "--prompts",
        "-p",
        type=str,
        nargs="+",
        help="Text prompts for Grounding DINO detection (e.g., --prompts dog person car)",
    )

    parser.add_argument(
        "--use-ai",
        action="store_true",
        help="Use AI-generated descriptions to create detection prompts",
    )

    parser.add_argument(
        "--vision-backend",
        type=str,
        default="openai",
        choices=["openai", "mistral"],
        help="Vision API for AI descriptions (default: openai)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # Configure logging
    import logging  # pylint: disable=import-outside-toplevel

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Validate input
    if not args.input.exists():
        logger.error("Input directory not found: %s", args.input)
        return 1

    # Import detection module
    try:
        # pylint: disable=import-outside-toplevel
        from dreamstack.raster.detection import (
            DetectionConfig,
            DetectionPipeline,
        )
    except ImportError as e:
        logger.error("Failed to import detection module: %s", e)
        logger.error(
            "Make sure you've installed the package or set PYTHONPATH"
        )
        return 1

    # Create configuration
    config = DetectionConfig(
        model_name=args.model,
        device=args.device,
        confidence_threshold=args.confidence,
        margin=args.margin,
        backend=args.backend,
        text_prompts=args.prompts,
        use_ai_description=args.use_ai,
        vision_backend=args.vision_backend,
    )

    logger.info("Configuration:")
    logger.info("  Backend: %s", config.backend)
    logger.info("  Model: %s", config.model_name)
    logger.info("  Device: %s", config.device)
    logger.info("  Confidence: %s", config.confidence_threshold)
    logger.info("  Margin: %spx", config.margin)
    if config.use_ai_description:
        logger.info("  AI Description: enabled (%s)", config.vision_backend)
    if config.text_prompts:
        logger.info("  Prompts: %s", ", ".join(config.text_prompts))

    # Create pipeline
    pipeline = DetectionPipeline(config)

    # Progress callback
    def progress(current: int, total: int, filename: str) -> None:
        print(f"[{current}/{total}] Processing: {filename}")

    # Process directory
    logger.info("Input: %s", args.input)
    logger.info("Output: %s", args.output)

    try:
        result = pipeline.process_directory(
            input_dir=args.input,
            output_dir=args.output,
            recursive=args.recursive,
            save_extracted=True,
            save_metadata_json=True,
            copy_source=args.copy_source,
            progress_callback=progress,
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Pipeline failed: %s", e)
        if args.verbose:
            import traceback  # pylint: disable=import-outside-toplevel

            traceback.print_exc()
        return 1

    # Print summary
    print()
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Images processed: {result.successful}/{result.total_images}")
    print(f"Objects extracted: {result.total_objects}")
    print(f"Failed: {result.failed}")

    if result.errors:
        print()
        print("Errors:")
        for path, error in result.errors.items():
            print(f"  {path}: {error}")

    print()
    print(f"Output saved to: {args.output}")

    return 0 if result.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
