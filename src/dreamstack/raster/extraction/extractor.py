# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Object Extractor
================

Main class for extracting objects from images.
Combines preprocessing, contour detection, and extraction
into a configurable, reusable interface.
"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Tuple

import cv2  # pylint: disable=no-member
import numpy as np
from numpy.typing import NDArray

from dreamstack.raster.analysis.analysis.analyzer import ColorAnalyzer
from dreamstack.raster.analysis.contour.detector import (
    ContourDetector,
    DetectionConfig,
)
from dreamstack.raster.analysis.contour.info import ContourInfo
from dreamstack.raster.analysis.preprocessing.processor import (
    ImagePreprocessor,
    PreprocessingConfig,
)
from dreamstack.raster.extraction.apply_background_mask import (
    apply_background_mask,
)
from dreamstack.raster.extraction.extract_object import extract_object
from dreamstack.raster.extraction.extract_with_alpha import extract_with_alpha


@dataclass
class ExtractedObject:
    """Represents an extracted object from an image.

    Attributes
    ----------
    image : NDArray[np.uint8]
        The extracted object image (BGR or BGRA).
    original_region : tuple[int, int, int, int]
        Region from original image (x, y, width, height).
    contour : ContourInfo | None
        Contour information used for extraction.
    index : int
        Sequential index of this object.
    source_path : Path | None
        Path to source image (if loaded from file).

    Examples
    --------
    >>> obj = ExtractedObject(image, (100, 100, 200, 200), contour, 0)
    >>> print(f"Object {obj.index}: {obj.dimensions}")
    """

    image: NDArray[np.uint8]
    original_region: Tuple[int, int, int, int]
    contour: ContourInfo | None = None
    index: int = 0
    source_path: Path | None = None

    @property
    def dimensions(self) -> Tuple[int, int]:
        """Get image dimensions as (height, width)."""
        return self.image.shape[:2]

    @property
    def area(self) -> float:
        """Get contour area, or 0 if no contour."""
        return self.contour.area if self.contour else 0.0

    @property
    def center(self) -> Tuple[float, float]:
        """Get center point in original image coordinates."""
        x, y, w, h = self.original_region
        return (x + w / 2, y + h / 2)


@dataclass
class ExtractionConfig:
    """Configuration for object extraction.

    Parameters
    ----------
    margin : int
        Margin around extracted objects. Default 25.
    min_dimension : int
        Minimum width/height for valid objects. Default 24.
    min_area_ratio : float
        Minimum object area as ratio of image. Default 0.0002.
    max_area_ratio : float
        Maximum object area as ratio of image. Default 0.95.
    target_size : int | None
        Target size for output images. None = no resize.
    with_alpha : bool
        Extract with transparent background. Default False.
    feather_edges : int
        Feathering for alpha edges. Default 0.
    preprocessing : PreprocessingConfig
        Preprocessing configuration.
    detection : DetectionConfig
        Contour detection configuration.

    Examples
    --------
    >>> config = ExtractionConfig(margin=50, with_alpha=True, feather_edges=3)
    >>> extractor = ObjectExtractor(config)
    """

    margin: int = 25
    min_dimension: int = 24
    min_area_ratio: float = 0.0002
    max_area_ratio: float = 0.95
    target_size: int | None = None
    with_alpha: bool = False
    feather_edges: int = 0
    preprocessing: PreprocessingConfig = field(
        default_factory=PreprocessingConfig
    )
    detection: DetectionConfig = field(default_factory=DetectionConfig)


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

    def extract(self, image: NDArray[np.uint8]) -> List[ExtractedObject]:
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
        if isinstance(processed, dict):
            binary = processed["threshold"]
        else:
            binary = processed

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
    ) -> List[ExtractedObject]:
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
            cv2.resize(  # pylint: disable=no-member
                image,
                (new_w, new_h),
                interpolation=cv2.INTER_AREA,  # pylint: disable=no-member
            ),
            dtype=np.uint8,
        )

    def extract_with_mask(
        self,
        image: NDArray[np.uint8],
        background_color: Tuple[int, int, int] | None = None,
    ) -> List[ExtractedObject]:
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
        objects: List[ExtractedObject],
        output_dir: str | Path,
        prefix: str = "object",
        extension: str = ".png",
    ) -> List[Path]:
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
            cv2.imwrite(str(filepath), obj.image)  # pylint: disable=no-member
            saved_paths.append(filepath)

        return saved_paths

    def with_config(self, **kwargs) -> "ObjectExtractor":
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
        # pylint: disable=import-outside-toplevel
        from dataclasses import (
            replace,
        )  # pylint: disable=import-outside-toplevel

        new_config = replace(self.config, **kwargs)
        return ObjectExtractor(new_config)
