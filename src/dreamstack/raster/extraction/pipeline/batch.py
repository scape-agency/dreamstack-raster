# -*- coding: utf-8 -*-

"""
Batch Pipeline
==============

High-level pipeline for batch object extraction with
configuration, progress tracking, and result aggregation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, List, Tuple

import cv2
from numpy.typing import NDArray

from dreamstack.raster.analysis.contour.detector import DetectionConfig
from dreamstack.raster.analysis.preprocessing.processor import PreprocessingConfig
from dreamstack.raster.extraction.extractor import (
    ExtractionConfig,
    ExtractedObject,
    ObjectExtractor,
)
from dreamstack.raster.extraction.pipeline.operations import (
    DEFAULT_IMAGE_FORMATS,
    find_images,
)

logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    """Results from batch processing.

    Attributes
    ----------
    source_path : Path
        Path to the source image.
    objects : list[ExtractedObject]
        Extracted objects from this image.
    output_paths : list[Path]
        Paths where objects were saved.
    error : str | None
        Error message if processing failed.

    Examples
    --------
    >>> result = BatchResult(path, objects, saved_paths)
    >>> print(f"{result.source_path.name}: {result.count} objects")
    """

    source_path: Path
    objects: List[ExtractedObject] = field(default_factory=list)
    output_paths: List[Path] = field(default_factory=list)
    error: str | None = None

    @property
    def success(self) -> bool:
        """Whether processing succeeded."""
        return self.error is None

    @property
    def count(self) -> int:
        """Number of extracted objects."""
        return len(self.objects)


@dataclass
class PipelineConfig:
    """Configuration for the extraction pipeline.

    Parameters
    ----------
    preprocessing : PreprocessingConfig
        Image preprocessing configuration.
    detection : DetectionConfig
        Contour detection configuration.
    extraction : ExtractionConfig
        Object extraction configuration.
    output_prefix : str
        Filename prefix for outputs. Default "object".
    output_extension : str
        Output file extension. Default ".png".
    file_formats : tuple
        Supported input file formats.
    save_intermediate : bool
        Save intermediate processing images. Default False.
    visualize_boxes : bool
        Create visualization with bounding boxes. Default False.
    box_color : tuple
        Bounding box color (B, G, R). Default green.
    box_thickness : int
        Bounding box line thickness. Default 2.

    Examples
    --------
    >>> config = PipelineConfig(
    ...     output_prefix="shell",
    ...     visualize_boxes=True,
    ... )
    >>> pipeline = BatchPipeline(config)
    """

    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    output_prefix: str = "object"
    output_extension: str = ".png"
    file_formats: tuple = DEFAULT_IMAGE_FORMATS
    save_intermediate: bool = False
    visualize_boxes: bool = False
    box_color: Tuple[int, int, int] = (36, 255, 12)
    box_thickness: int = 2


class BatchPipeline:
    """High-level pipeline for batch object extraction.

    Provides a convenient interface for processing multiple images
    with progress tracking, error handling, and result aggregation.

    Attributes
    ----------
    config : PipelineConfig
        Pipeline configuration.
    extractor : ObjectExtractor
        Underlying object extractor.

    Examples
    --------
    >>> from dreamstack.raster.extraction.pipeline import BatchPipeline
    >>> pipeline = BatchPipeline()
    >>>
    >>> # Process single image
    >>> result = pipeline.process_image("scan.jpg", "output/")
    >>> print(f"Extracted {result.count} objects")
    >>>
    >>> # Process directory with progress
    >>> def on_progress(current, total, name):
    ...     print(f"[{current}/{total}] {name}")
    >>>
    >>> results = pipeline.process_directory(
    ...     "scans/",
    ...     "output/",
    ...     progress_callback=on_progress,
    ... )

    Notes
    -----
    The pipeline builds on ObjectExtractor with additional features
    for batch processing, progress tracking, and visualization.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        """Initialize the pipeline.

        Parameters
        ----------
        config : PipelineConfig | None, optional
            Pipeline configuration. Uses defaults if None.
        """
        self.config = config or PipelineConfig()

        # Build extraction config from pipeline config
        extraction_config = ExtractionConfig(
            margin=self.config.extraction.margin,
            min_dimension=self.config.extraction.min_dimension,
            min_area_ratio=self.config.extraction.min_area_ratio,
            max_area_ratio=self.config.extraction.max_area_ratio,
            target_size=self.config.extraction.target_size,
            with_alpha=self.config.extraction.with_alpha,
            feather_edges=self.config.extraction.feather_edges,
            preprocessing=self.config.preprocessing,
            detection=self.config.detection,
        )

        self.extractor = ObjectExtractor(extraction_config)

    def process_image(
        self,
        image_path: str | Path,
        output_dir: str | Path | None = None,
    ) -> BatchResult:
        """Process a single image and extract objects.

        Parameters
        ----------
        image_path : str | Path
            Path to the input image.
        output_dir : str | Path | None, optional
            Directory to save outputs. None = don't save.

        Returns
        -------
        BatchResult
            Processing result with extracted objects.
        """
        image_path = Path(image_path)
        logger.info(f"Processing: {image_path.name}")

        try:
            objects = self.extractor.extract_from_file(image_path)
            logger.info(f"Found {len(objects)} objects")

            output_paths = []
            if output_dir:
                output_dir = Path(output_dir)
                prefix = f"{self.config.output_prefix}_{image_path.stem}"
                output_paths = self.extractor.save_objects(
                    objects,
                    output_dir,
                    prefix=prefix,
                    extension=self.config.output_extension,
                )
                logger.info(f"Saved {len(output_paths)} objects")

                # Save visualization if requested
                if self.config.visualize_boxes:
                    self._save_visualization(image_path, objects, output_dir)

            return BatchResult(
                source_path=image_path,
                objects=objects,
                output_paths=output_paths,
            )

        except Exception as e:
            logger.error(f"Failed to process {image_path}: {e}")
            return BatchResult(
                source_path=image_path,
                error=str(e),
            )

    def process_directory(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        recursive: bool = False,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> List[BatchResult]:
        """Process all images in a directory.

        Parameters
        ----------
        input_dir : str | Path
            Input directory containing images.
        output_dir : str | Path
            Output directory for extracted objects.
        recursive : bool, optional
            Process subdirectories. Default False.
        progress_callback : callable | None, optional
            Progress callback: callback(current, total, filename).

        Returns
        -------
        list[BatchResult]
            Results for each processed image.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        image_files = find_images(
            input_dir, self.config.file_formats, recursive
        )

        total = len(image_files)
        logger.info(f"Found {total} images to process")

        results = []
        for idx, image_path in enumerate(image_files, 1):
            if progress_callback:
                progress_callback(idx, total, image_path.name)

            result = self.process_image(image_path, output_dir)
            results.append(result)

        # Summary
        total_objects = sum(r.count for r in results)
        failed = sum(1 for r in results if not r.success)
        logger.info(
            f"Completed: {total_objects} objects from "
            f"{total - failed}/{total} images"
        )

        return results

    def process_directory_iter(
        self,
        input_dir: str | Path,
        output_dir: str | Path | None = None,
        recursive: bool = False,
    ) -> Iterator[BatchResult]:
        """Process directory and yield results incrementally.

        Parameters
        ----------
        input_dir : str | Path
            Input directory.
        output_dir : str | Path | None, optional
            Output directory.
        recursive : bool, optional
            Process subdirectories.

        Yields
        ------
        BatchResult
            Result for each processed image.
        """
        input_dir = Path(input_dir)
        image_files = find_images(
            input_dir, self.config.file_formats, recursive
        )

        for image_path in image_files:
            yield self.process_image(image_path, output_dir)

    def visualize(
        self,
        image_path: str | Path,
        output_path: str | Path | None = None,
    ) -> NDArray:
        """Create visualization with bounding boxes.

        Parameters
        ----------
        image_path : str | Path
            Input image path.
        output_path : str | Path | None, optional
            Path to save visualization. None = don't save.

        Returns
        -------
        NDArray
            Image with bounding boxes drawn.
        """
        image_path = Path(image_path)
        image = self.extractor.preprocessor.load(image_path)
        objects = self.extractor.extract(image)

        return self._draw_boxes(image, objects, output_path)

    def _draw_boxes(
        self,
        image: NDArray,
        objects: List[ExtractedObject],
        output_path: str | Path | None = None,
    ) -> NDArray:
        """Draw bounding boxes on image.

        Parameters
        ----------
        image : NDArray
            Source image.
        objects : list[ExtractedObject]
            Extracted objects.
        output_path : str | Path | None, optional
            Path to save image.

        Returns
        -------
        NDArray
            Image with boxes drawn.
        """
        annotated = image.copy()

        for obj in objects:
            x, y, w, h = obj.original_region
            cv2.rectangle(
                annotated,
                (x, y),
                (x + w, y + h),
                self.config.box_color,
                self.config.box_thickness,
            )

        if output_path:
            cv2.imwrite(str(output_path), annotated)

        return annotated

    def _save_visualization(
        self,
        image_path: Path,
        objects: List[ExtractedObject],
        output_dir: Path,
    ) -> None:
        """Save visualization image.

        Parameters
        ----------
        image_path : Path
            Source image path.
        objects : list[ExtractedObject]
            Extracted objects.
        output_dir : Path
            Output directory.
        """
        image = self.extractor.preprocessor.load(image_path)
        output_path = output_dir / f"{image_path.stem}_boxes.jpg"
        self._draw_boxes(image, objects, output_path)

    def with_config(self, **kwargs) -> "BatchPipeline":
        """Create a new pipeline with modified configuration.

        Parameters
        ----------
        **kwargs
            Configuration parameters to override.

        Returns
        -------
        BatchPipeline
            New pipeline with updated config.
        """
        from dataclasses import replace
        new_config = replace(self.config, **kwargs)
        return BatchPipeline(new_config)

    @staticmethod
    def quick_extract(
        image_path: str | Path,
        output_dir: str | Path,
    ) -> List[ExtractedObject]:
        """Quick extraction with default settings.

        Convenience method for simple extraction tasks.

        Parameters
        ----------
        image_path : str | Path
            Input image.
        output_dir : str | Path
            Output directory.

        Returns
        -------
        list[ExtractedObject]
            Extracted objects.

        Examples
        --------
        >>> objects = BatchPipeline.quick_extract("scan.jpg", "output/")
        """
        pipeline = BatchPipeline()
        result = pipeline.process_image(image_path, output_dir)
        return result.objects
