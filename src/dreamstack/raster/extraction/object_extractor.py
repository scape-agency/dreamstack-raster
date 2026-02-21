"""
Object Extractor
================

Main class for extracting objects from images.
Combines preprocessing, contour detection, and extraction
into a configurable, reusable interface.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from dreamstack.raster.analysis.analysis.analyzer import ColorAnalyzer
from dreamstack.raster.analysis.contour.detector import ContourDetector
from dreamstack.raster.analysis.contour.info import ContourInfo
from dreamstack.raster.analysis.preprocessing.processor import (
    ImagePreprocessor,
)
from dreamstack.raster.extraction.apply_background_mask import (
    apply_background_mask,
)
from dreamstack.raster.extraction.extract_object import extract_object
from dreamstack.raster.extraction.extract_with_alpha import extract_with_alpha
from dreamstack.raster.extraction.extracted_object import ExtractedObject
from dreamstack.raster.extraction.extraction_config import ExtractionConfig


class ObjectExtractor:
    """Extracts individual objects from images.

    Combines preprocessing, contour detection, and extraction
    to identify and extract individual objects from larger images.
    Designed to be reusable and configurable for various use cases.

    Attributes
    ----------
    config : ExtractionConfig
        Extraction configuration.
    preprocessor : ImagePreprocessor
        Image preprocessing component.
    detector : ContourDetector
        Contour detection component.
    color_analyzer : ColorAnalyzer
        Color analysis component.

    Examples
    --------
    >>> from dreamstack.raster.extraction import ObjectExtractor
    >>> extractor = ObjectExtractor()
    >>>
    >>> # Extract from file
    >>> objects = extractor.extract_from_file("scan.jpg")
    >>> for obj in objects:
    ...     cv2.imwrite(f"object_{obj.index}.png", obj.image)
    >>>
    >>> # Extract with alpha channel
    >>> config = ExtractionConfig(with_alpha=True, feather_edges=2)
    >>> extractor = ObjectExtractor(config)
    >>> objects = extractor.extract(image)

    Notes
    -----
    The extractor is designed to be reusable across multiple images
    with consistent configuration. Create different extractor instances
    for different types of content (e.g., shells vs. documents).
    """

    def __init__(self, config: ExtractionConfig | None = None) -> None:
        """Initialize the object extractor.

        Parameters
        ----------
        config : ExtractionConfig | None, optional
            Extraction configuration. Uses defaults if None.
        """
        self.config = config or ExtractionConfig()
        self.preprocessor = ImagePreprocessor(self.config.preprocessing)
        self.detector = ContourDetector(self.config.detection)
        self.color_analyzer = ColorAnalyzer()

    def extract(self, image: NDArray[np.uint8]) -> list[ExtractedObject]:
        """Extract all objects from an image.

        Parameters
        ----------
        image : NDArray[np.uint8]
            Input image in BGR format.

        Returns
        -------
        list[ExtractedObject]
            List of extracted objects, sorted by area (largest first).

        Examples
        --------
        >>> objects = extractor.extract(image)
        >>> print(f"Found {len(objects)} objects")
        """
        return list(self.extract_iter(image))

    def extract_iter(
        self,
        image: NDArray[np.uint8],
    ) -> Iterator[ExtractedObject]:
        """Extract objects as an iterator.

        Useful for processing large images where you want to handle
        each object as it's extracted.

        Parameters
        ----------
        image : NDArray[np.uint8]
            Input image in BGR format.

        Yields
        ------
        ExtractedObject
            Extracted objects one at a time.
        """
        # Preprocess
        processed = self.preprocessor.preprocess(image)
        assert isinstance(processed, dict), "Expected dict from preprocess()"
        binary = processed["threshold"]

        # Detect contours
        contours = self.detector.detect(binary)

        # Filter by area
        h, w = image.shape[:2]
        image_area = h * w
        contours = [
            c
            for c in contours
            if image_area * self.config.min_area_ratio
            <= c.area
            <= image_area * self.config.max_area_ratio
        ]

        # Extract each object
        for idx, contour in enumerate(contours):
            obj = self._extract_single(image, contour, idx)
            if obj is not None:
                yield obj

    def extract_from_file(
        self,
        path: str | Path,
    ) -> list[ExtractedObject]:
        """Extract objects from an image file.

        Parameters
        ----------
        path : str | Path
            Path to the input image.

        Returns
        -------
        list[ExtractedObject]
            List of extracted objects.
        """
        path = Path(path)
        image = self.preprocessor.load(path)
        objects = self.extract(image)

        # Add source path to objects
        for obj in objects:
            obj.source_path = path

        return objects

    def _extract_single(
        self,
        image: NDArray[np.uint8],
        contour: ContourInfo,
        index: int,
    ) -> ExtractedObject | None:
        """Extract a single object from the image.

        Parameters
        ----------
        image : NDArray[np.uint8]
            Source image.
        contour : ContourInfo
            Contour information.
        index : int
            Object index.

        Returns
        -------
        ExtractedObject | None
            Extracted object or None if invalid.
        """
        if self.config.with_alpha:
            obj_image = extract_with_alpha(
                image,
                contour,
                margin=self.config.margin,
                feather=self.config.feather_edges,
            )
        else:
            obj_image = extract_object(
                image,
                contour,
                margin=self.config.margin,
                min_dimension=self.config.min_dimension,
            )

        if obj_image is None:
            return None

        # Resize if needed
        if self.config.target_size:
            obj_image = self._resize_to_target(obj_image)

        # Calculate original region
        x, y, w, h = contour.bounding_rect
        margin = self.config.margin
        region = (
            max(0, x - margin),
            max(0, y - margin),
            w + margin * 2,
            h + margin * 2,
        )

        return ExtractedObject(
            image=obj_image,
            original_region=region,
            contour=contour,
            index=index,
        )

    def _resize_to_target(
        self,
        image: NDArray[np.uint8],
    ) -> NDArray[np.uint8]:
        """Resize image to target size maintaining aspect ratio.

        Parameters
        ----------
        image : NDArray[np.uint8]
            Input image.

        Returns
        -------
        NDArray[np.uint8]
            Resized image.
        """
        if self.config.target_size is None:
            return image

        h, w = image.shape[:2]
        target = self.config.target_size

        # Calculate scale
        scale = target / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)

        return np.asarray(
            cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA),
            dtype=np.uint8,
        )

    def extract_with_mask(
        self,
        image: NDArray[np.uint8],
        background_color: tuple[int, int, int] | None = None,
    ) -> list[ExtractedObject]:
        """Extract objects with masked backgrounds.

        Extracts objects and replaces their backgrounds with a solid
        color based on the dominant background color.

        Parameters
        ----------
        image : NDArray[np.uint8]
            Input image in BGR format.
        background_color : tuple[int, int, int] | None, optional
            Background color (B, G, R). If None, auto-detected.

        Returns
        -------
        list[ExtractedObject]
            Extracted objects with masked backgrounds.
        """
        if background_color is None:
            bg = self.color_analyzer.background_color(image)
            background_color = tuple(bg)

        objects = self.extract(image)
        masked_objects = []

        for obj in objects:
            masked_image = apply_background_mask(
                obj.image,
                background_color,
            )
            masked_objects.append(
                ExtractedObject(
                    image=masked_image,
                    original_region=obj.original_region,
                    contour=obj.contour,
                    index=obj.index,
                    source_path=obj.source_path,
                )
            )

        return masked_objects

    def save_objects(
        self,
        objects: list[ExtractedObject],
        output_dir: str | Path,
        prefix: str = "object",
        extension: str = ".png",
    ) -> list[Path]:
        """Save extracted objects to disk.

        Parameters
        ----------
        objects : list[ExtractedObject]
            Objects to save.
        output_dir : str | Path
            Output directory path.
        prefix : str, optional
            Filename prefix. Default "object".
        extension : str, optional
            File extension. Default ".png".

        Returns
        -------
        list[Path]
            List of saved file paths.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        for obj in objects:
            filename = f"{prefix}_{obj.index:04d}{extension}"
            filepath = output_dir / filename
            cv2.imwrite(str(filepath), obj.image)
            saved_paths.append(filepath)

        return saved_paths

    def with_config(self, **kwargs) -> ObjectExtractor:
        """Create a new extractor with modified configuration.

        Parameters
        ----------
        **kwargs
            Configuration parameters to override.

        Returns
        -------
        ObjectExtractor
            New extractor with updated config.

        Examples
        --------
        >>> extractor = ObjectExtractor()
        >>> alpha_extractor = extractor.with_config(with_alpha=True)
        """
        from dataclasses import (
            replace,
        )  # pylint: disable=import-outside-toplevel

        new_config = replace(self.config, **kwargs)
        return ObjectExtractor(new_config)
