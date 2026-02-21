"""
Detection Extractor
===================

Extract detected objects from images with segmentation masks.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from dreamstack.raster.detection.config import DetectionConfig
from dreamstack.raster.detection.factory import create_detector
from dreamstack.raster.detection.result import (
    DetectionResult,
    ImageDetectionResult,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from dreamstack.raster.detection.base import BaseDetector

logger = logging.getLogger(__name__)


class ExtractedDetection:
    """An extracted object from detection.

    Attributes
    ----------
    image : NDArray[np.uint8]
        Extracted image (BGRA with alpha if mask available).
    detection : DetectionResult
        Original detection result.
    filename : str
        Suggested filename for saving.
    """

    def __init__(
        self,
        image: NDArray[np.uint8],
        detection: DetectionResult,
        filename: str,
    ) -> None:
        self.image = image
        self.detection = detection
        self.filename = filename

    @property
    def label(self) -> str:
        """Object label."""
        return self.detection.label

    @property
    def confidence(self) -> float:
        """Detection confidence."""
        return self.detection.confidence

    @property
    def has_alpha(self) -> bool:
        """Whether image has alpha channel."""
        return self.image.shape[2] == 4 if len(self.image.shape) > 2 else False


class DetectionExtractor:
    """Extract detected objects from images.

    Combines object detection with extraction, applying
    segmentation masks as alpha channels for clean cutouts.

    Example
    -------
    >>> from dreamstack.raster.detection import DetectionExtractor, DetectionConfig
    >>>
    >>> extractor = DetectionExtractor()
    >>> extractions = extractor.extract(image)
    >>> for ext in extractions:
    ...     cv2.imwrite(ext.filename, ext.image)
    """

    def __init__(
        self,
        config: DetectionConfig | None = None,
        detector: BaseDetector | None = None,
    ) -> None:
        """Initialize extractor.

        Parameters
        ----------
        config : DetectionConfig | None
            Detection configuration. Uses defaults if None.
        detector : BaseDetector | None
            Pre-configured detector. Created from config if None.
        """
        self.config = config or DetectionConfig()
        self._detector = detector

    @property
    def detector(self) -> BaseDetector:
        """Get or create detector instance."""
        if self._detector is None:
            self._detector = create_detector(self.config)
        return self._detector

    def detect(self, image: NDArray[np.uint8]) -> ImageDetectionResult:
        """Run detection on an image.

        Parameters
        ----------
        image : NDArray[np.uint8]
            Input image (BGR).

        Returns
        -------
        ImageDetectionResult
            Detection results.
        """
        return self.detector.detect(image)

    def extract(
        self,
        image: NDArray[np.uint8],
        source_path: Path | None = None,
    ) -> list[ExtractedDetection]:
        """Detect and extract all objects from an image.

        Parameters
        ----------
        image : NDArray[np.uint8]
            Input image (BGR).
        source_path : Path | None
            Source path for naming context.

        Returns
        -------
        list[ExtractedDetection]
            List of extracted objects with filenames.
        """
        result = self.detect(image)
        result.source_path = source_path

        extractions: list[ExtractedDetection] = []
        label_counts: dict[str, int] = {}

        for detection in result.detections:
            # Generate unique filename
            label = detection.label.replace(" ", "_").lower()
            count = label_counts.get(label, 0) + 1
            label_counts[label] = count
            filename = f"{label}_{count}.png"

            # Extract region
            extracted = self._extract_detection(image, detection)
            if extracted is not None:
                extractions.append(
                    ExtractedDetection(
                        image=extracted,
                        detection=detection,
                        filename=filename,
                    )
                )

        return extractions

    def _extract_detection(
        self,
        image: NDArray[np.uint8],
        detection: DetectionResult,
    ) -> NDArray[np.uint8] | None:
        """Extract a single detection from the image.

        Parameters
        ----------
        image : NDArray[np.uint8]
            Source image.
        detection : DetectionResult
            Detection to extract.

        Returns
        -------
        NDArray[np.uint8] | None
            Extracted image with alpha channel, or None if invalid.
        """
        x, y, w, h = detection.bbox
        margin = self.config.margin
        img_h, img_w = image.shape[:2]

        # Apply margin with bounds checking
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(img_w, x + w + margin)
        y2 = min(img_h, y + h + margin)

        # Extract region
        region = image[y1:y2, x1:x2].copy()

        if region.size == 0:
            return None

        # Validate dimensions
        region_h, region_w = region.shape[:2]
        if (
            region_w < self.config.min_dimension
            or region_h < self.config.min_dimension
        ):
            return None

        # Apply mask as alpha if available
        if (
            detection.has_mask
            and detection.mask is not None
            and self.config.segmentation.enabled
        ):
            region = self._apply_mask_alpha(region, detection.mask, margin)

        return region

    def _apply_mask_alpha(
        self,
        region: NDArray[np.uint8],
        mask: NDArray[np.uint8],
        margin: int,
    ) -> NDArray[np.uint8]:
        """Apply segmentation mask as alpha channel.

        Parameters
        ----------
        region : NDArray[np.uint8]
            Extracted region (BGR).
        mask : NDArray[np.uint8]
            Segmentation mask for the detection.
        margin : int
            Margin that was applied to the region.

        Returns
        -------
        NDArray[np.uint8]
            Region with alpha channel (BGRA).
        """
        region_h, region_w = region.shape[:2]
        mask_h, mask_w = mask.shape[:2]

        # Create full-size alpha channel
        alpha = np.zeros((region_h, region_w), dtype=np.uint8)

        # Calculate mask placement (accounting for margin)
        paste_x = margin
        paste_y = margin

        # Handle boundary cases
        src_x1 = max(0, -paste_x)
        src_y1 = max(0, -paste_y)
        src_x2 = min(mask_w, region_w - paste_x)
        src_y2 = min(mask_h, region_h - paste_y)

        dst_x1 = max(0, paste_x)
        dst_y1 = max(0, paste_y)
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)

        # Copy mask into alpha
        if src_x2 > src_x1 and src_y2 > src_y1:
            alpha[dst_y1:dst_y2, dst_x1:dst_x2] = mask[
                src_y1:src_y2, src_x1:src_x2
            ]

        # Convert BGR to BGRA
        if len(region.shape) == 2:
            region = cv2.cvtColor(region, cv2.COLOR_GRAY2BGR)  # type: ignore[assignment]

        bgra = cv2.cvtColor(region, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = alpha

        return bgra

    def extract_from_file(
        self,
        path: str | Path,
    ) -> list[ExtractedDetection]:
        """Load image and extract all objects.

        Parameters
        ----------
        path : str | Path
            Path to image file.

        Returns
        -------
        list[ExtractedDetection]
            List of extracted objects.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Failed to load image: {path}")

        return self.extract(image, source_path=path)
