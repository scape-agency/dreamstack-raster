# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Detection Pipeline
==================

Batch processing pipeline for object detection and extraction.
"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

import logging
import shutil
from collections.abc import Callable

# ThreadPoolExecutor reserved for future parallel processing
# from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from dreamstack.raster.detection.config import DetectionConfig
from dreamstack.raster.detection.extractor import (
    DetectionExtractor,
    ExtractedDetection,
)
from dreamstack.raster.detection.metadata import (
    ImageMetadata,
    create_image_metadata,
    save_metadata,
)

logger = logging.getLogger(__name__)


# Supported image extensions
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tiff",
    ".tif",
    ".webp",
    ".gif",
}


@dataclass
class PipelineResult:
    """Results from pipeline processing.

    Attributes
    ----------
    total_images : int
        Total images processed.
    total_objects : int
        Total objects extracted.
    successful : int
        Successfully processed images.
    failed : int
        Failed images.
    errors : dict[str, str]
        Mapping of failed paths to error messages.
    """

    total_images: int = 0
    total_objects: int = 0
    successful: int = 0
    failed: int = 0
    errors: dict[str, str] = field(default_factory=dict)


ProgressCallback = Callable[[int, int, str], None]


class DetectionPipeline:
    """Batch detection and extraction pipeline.

    Processes a directory of images, detecting objects,
    extracting them with segmentation masks, and generating
    metadata JSON files.

    Example
    -------
    >>> from dreamstack.raster.detection import DetectionPipeline, DetectionConfig
    >>>
    >>> config = DetectionConfig(
    ...     model_name="yolov8n-seg",
    ...     confidence_threshold=0.5,
    ... )
    >>> pipeline = DetectionPipeline(config)
    >>>
    >>> results = pipeline.process_directory(
    ...     input_dir="./images",
    ...     output_dir="./output",
    ... )
    >>> print(f"Extracted {results.total_objects} objects from {results.total_images} images")

    Output Structure
    ----------------
    output/
    ├── image_0001/
    │   ├── metadata.json
    │   ├── dog_1.png
    │   ├── person_1.png
    │   └── bicycle_1.png
    ├── image_0002/
    │   ├── metadata.json
    │   └── car_1.png
    └── ...
    """

    def __init__(
        self,
        config: DetectionConfig | None = None,
        max_workers: int = 1,
    ) -> None:
        """Initialize pipeline.

        Parameters
        ----------
        config : DetectionConfig | None
            Detection configuration. Uses defaults if None.
        max_workers : int
            Maximum parallel workers. Default 1 (sequential).
            Note: YOLO inference should be sequential for GPU.
        """
        self.config = config or DetectionConfig()
        self.max_workers = max_workers
        self._extractor = DetectionExtractor(self.config)
        self._describer = None

        # Initialize AI describer if enabled
        if self.config.use_ai_description:
            self._init_describer()

    def _init_describer(self) -> None:
        """Initialize AI image describer."""
        # pylint: disable=import-outside-toplevel
        from dreamstack.raster.detection.describer import (
            DescriptionConfig,
            ImageDescriber,
        )

        desc_config = DescriptionConfig(
            backend=self.config.vision_backend,
        )
        self._describer = ImageDescriber(desc_config)
        logger.info(
            "AI description enabled with %s", self.config.vision_backend
        )

    def _get_ai_prompts(self, image_path: Path) -> list[str]:
        """Get detection prompts from AI image description.

        Parameters
        ----------
        image_path : Path
            Path to image file.

        Returns
        -------
        list[str]
            List of object names to detect.
        """
        if self._describer is None:
            return []

        try:
            result = self._describer.describe(image_path)
            logger.info("AI detected objects: %s", result.objects)
            return result.objects
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning("AI description failed: %s", e)
            return []

    def process_directory(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        *,
        recursive: bool = False,
        save_extracted: bool = True,
        save_metadata_json: bool = True,
        copy_source: bool = False,
        progress_callback: ProgressCallback | None = None,
    ) -> PipelineResult:
        """Process all images in a directory.

        Parameters
        ----------
        input_dir : str | Path
            Input directory containing images.
        output_dir : str | Path
            Output directory for results.
        recursive : bool
            Process subdirectories recursively.
        save_extracted : bool
            Save extracted object images.
        save_metadata_json : bool
            Save metadata.json per image.
        copy_source : bool
            Copy source image to output folder.
        progress_callback : ProgressCallback | None
            Callback for progress updates: (current, total, filename).

        Returns
        -------
        PipelineResult
            Processing statistics.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        # Find all images
        images = self._find_images(input_dir, recursive)
        total = len(images)

        if total == 0:
            logger.warning("No images found in %s", input_dir)
            return PipelineResult()

        logger.info("Found %d images to process", total)

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        result = PipelineResult(total_images=total)

        # Process images (sequential for GPU inference)
        for i, image_path in enumerate(images):
            try:
                if progress_callback:
                    progress_callback(i + 1, total, image_path.name)

                num_objects = self._process_image(
                    image_path,
                    output_dir,
                    input_dir=input_dir,
                    save_extracted=save_extracted,
                    save_metadata_json=save_metadata_json,
                    copy_source=copy_source,
                )

                result.total_objects += num_objects
                result.successful += 1

            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.error("Failed to process %s: %s", image_path, e)
                result.failed += 1
                result.errors[str(image_path)] = str(e)

        return result

    def _find_images(
        self,
        directory: Path,
        recursive: bool,
    ) -> list[Path]:
        """Find all image files in directory.

        Parameters
        ----------
        directory : Path
            Directory to search.
        recursive : bool
            Search recursively.

        Returns
        -------
        list[Path]
            List of image file paths.
        """
        pattern = "**/*" if recursive else "*"
        images = []

        for path in directory.glob(pattern):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(path)

        return sorted(images)

    def _process_image(
        self,
        image_path: Path,
        output_dir: Path,
        *,
        input_dir: Path | None = None,
        save_extracted: bool,
        save_metadata_json: bool,
        copy_source: bool,
    ) -> int:
        """Process a single image.

        Parameters
        ----------
        image_path : Path
            Path to image file.
        output_dir : Path
            Output directory.
        input_dir : Path | None
            Input directory root (for preserving folder structure).
        save_extracted : bool
            Save extracted images.
        save_metadata_json : bool
            Save metadata JSON.
        copy_source : bool
            Copy source image.

        Returns
        -------
        int
            Number of objects extracted.
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Create output folder preserving input structure
        if input_dir is not None:
            # Get relative path from input_dir to image's parent
            try:
                relative_parent = image_path.parent.relative_to(input_dir)
                image_output_dir = (
                    output_dir / relative_parent / image_path.stem
                )
            except ValueError:
                # Fallback if not relative
                image_output_dir = output_dir / image_path.stem
        else:
            image_output_dir = output_dir / image_path.stem
        image_output_dir.mkdir(parents=True, exist_ok=True)

        # Get AI-generated prompts if enabled
        ai_description = None
        if self.config.use_ai_description and self._describer is not None:
            try:
                ai_description = self._describer.describe(image_path)
                ai_prompts = ai_description.objects

                # Update detector prompts for Grounding DINO
                if (
                    self.config.backend == "grounding_dino_sam"
                    and hasattr(self._extractor.detector, "set_prompts")
                    and ai_prompts
                ):
                    self._extractor.detector.set_prompts(ai_prompts)  # type: ignore[attr-defined]
                    logger.info(
                        "AI prompts: %s...", ", ".join(ai_prompts[:10])
                    )
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.warning(
                    "AI description failed for %s: %s", image_path.name, e
                )

        # Run detection and extraction
        extractions = self._extractor.extract(image, source_path=image_path)

        logger.info(
            "Detected %d objects in %s", len(extractions), image_path.name
        )

        # Save extracted objects
        if save_extracted:
            for ext in extractions:
                output_path = image_output_dir / ext.filename
                cv2.imwrite(str(output_path), ext.image)

        # Save metadata
        if save_metadata_json:
            metadata = create_image_metadata(
                source_path=image_path,
                image_size=image.shape[:2],
                extractions=extractions,
                ai_description=(
                    ai_description.description if ai_description else None
                ),
            )
            save_metadata(metadata, image_output_dir / "metadata.json")

        # Copy source image
        if copy_source:
            shutil.copy2(image_path, image_output_dir / image_path.name)

        return len(extractions)

    def process_single(
        self,
        image_path: str | Path,
        output_dir: str | Path,
    ) -> tuple[list[ExtractedDetection], ImageMetadata]:
        """Process a single image.

        Parameters
        ----------
        image_path : str | Path
            Path to image file.
        output_dir : str | Path
            Output directory.

        Returns
        -------
        tuple[list[ExtractedDetection], ImageMetadata]
            Extractions and metadata.
        """
        image_path = Path(image_path)
        output_dir = Path(output_dir)

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        extractions = self._extractor.extract(image, source_path=image_path)

        metadata = create_image_metadata(
            source_path=image_path,
            image_size=image.shape[:2],
            extractions=extractions,
            ai_description=None,  # Single image processing doesn't use AI description
        )

        # Save
        image_output_dir = output_dir / image_path.stem
        image_output_dir.mkdir(parents=True, exist_ok=True)

        for ext in extractions:
            cv2.imwrite(str(image_output_dir / ext.filename), ext.image)

        save_metadata(metadata, image_output_dir / "metadata.json")

        return extractions, metadata
