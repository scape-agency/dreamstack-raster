"""
Ultralytics YOLO Backend
========================

Object detection using Ultralytics YOLO models.
Optimized for Mac M2 with MPS support.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from dreamstack.raster.detection.base import BaseDetector
from dreamstack.raster.detection.config import DetectionConfig
from dreamstack.raster.detection.result import (
    DetectionResult,
    ImageDetectionResult,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class UltralyticsDetector(BaseDetector):
    """YOLO-based object detection using Ultralytics.

    Supports YOLOv8/v11 models with instance segmentation.
    Optimized for Mac M2 with automatic MPS detection.

    Example
    -------
    >>> from dreamstack.raster.detection import DetectionConfig
    >>> from dreamstack.raster.detection.backends import UltralyticsDetector
    >>>
    >>> config = DetectionConfig(model_name="yolov8n-seg")
    >>> detector = UltralyticsDetector(config)
    >>> result = detector.detect(image)
    >>> for det in result.detections:
    ...     print(f"{det.label}: {det.confidence:.2f}")

    Notes
    -----
    Requires: pip install ultralytics

    Model variants:
    - yolov8n-seg: Nano (fastest, ~3MB)
    - yolov8s-seg: Small (~11MB)
    - yolov8m-seg: Medium (~26MB)
    - yolov8l-seg: Large (~46MB)
    - yolov8x-seg: Extra large (~69MB)
    """

    def __init__(self, config: DetectionConfig | None = None) -> None:
        """Initialize YOLO detector.

        Parameters
        ----------
        config : DetectionConfig | None
            Detection configuration. Uses defaults if None.
        """
        super().__init__(config or DetectionConfig())
        self._class_names: dict[int, str] = {}

    def _load_model(self) -> None:
        """Load YOLO model with lazy import."""
        try:
            # pylint: disable=import-outside-toplevel
            from ultralytics import YOLO  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required for YOLO detection. "
                "Install with: pip install ultralytics"
            ) from exc

        device = self._resolve_device()
        logger.info(
            "Loading YOLO model '%s' on %s", self.config.model_name, device
        )

        # Load model - will download if not present
        self._model = YOLO(self.config.model_name)

        # Move to device
        if device != "cpu":
            self._model.to(device)

        # Cache class names
        self._class_names = self._model.names
        self._model_loaded = True

        logger.info(
            "Model loaded: %d classes, device=%s",
            len(self._class_names),
            device,
        )

    def get_class_names(self) -> dict[int, str]:
        """Get COCO class names mapping.

        Returns
        -------
        dict[int, str]
            Mapping from class ID to class name.
        """
        self._ensure_model_loaded()
        return self._class_names.copy()

    def detect(self, image: NDArray[np.uint8]) -> ImageDetectionResult:
        """Run object detection on an image.

        Parameters
        ----------
        image : NDArray[np.uint8]
            Input image (BGR format, HxWxC).

        Returns
        -------
        ImageDetectionResult
            Detection results with bounding boxes, labels, and masks.
        """
        self._ensure_model_loaded()

        # Get image dimensions
        h, w = image.shape[:2]

        # Run inference
        results = self._model.predict(
            image,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            max_det=self.config.max_detections,
            classes=self.config.classes,
            verbose=False,
        )

        # Parse results
        detections: list[DetectionResult] = []

        if results and len(results) > 0:
            result = results[0]  # Single image

            # Get boxes
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                # Get masks if available (for -seg models)
                masks = result.masks
                has_masks = masks is not None and len(masks) > 0

                for i, box in enumerate(boxes):
                    # Extract box data
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, xyxy)
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1

                    # Skip tiny detections
                    if (
                        bbox_width < self.config.min_dimension
                        or bbox_height < self.config.min_dimension
                    ):
                        continue

                    class_id = int(box.cls[0].cpu().numpy())
                    confidence = float(box.conf[0].cpu().numpy())
                    label = self._class_names.get(
                        class_id, f"class_{class_id}"
                    )

                    # Extract mask for this detection
                    mask = None
                    if has_masks and self.config.segmentation.enabled:
                        mask = self._extract_mask(
                            masks.data[i].cpu().numpy(),  # type: ignore[union-attr]
                            (h, w),
                            (x1, y1, x2, y2),
                        )

                    detections.append(
                        DetectionResult(
                            label=label,
                            class_id=class_id,
                            confidence=confidence,
                            bbox=(x1, y1, bbox_width, bbox_height),
                            mask=mask,
                        )
                    )

        return ImageDetectionResult(
            source_path=None,
            image_size=(h, w),
            detections=detections,
        )

    def _extract_mask(
        self,
        mask_data: NDArray[np.float32],
        image_size: tuple[int, int],
        bbox: tuple[int, int, int, int],
    ) -> NDArray[np.uint8]:
        """Extract and crop mask to bounding box region.

        Parameters
        ----------
        mask_data : NDArray[np.float32]
            Raw mask from YOLO (may be different resolution).
        image_size : tuple[int, int]
            Original image size (h, w).
        bbox : tuple[int, int, int, int]
            Bounding box (x1, y1, x2, y2).

        Returns
        -------
        NDArray[np.uint8]
            Cropped binary mask (0-255).
        """
        import cv2  # pylint: disable=import-outside-toplevel

        h, w = image_size
        x1, y1, x2, y2 = bbox

        # Resize mask to image size if needed
        if mask_data.shape != image_size:
            mask_resized = cv2.resize(
                mask_data, (w, h), interpolation=cv2.INTER_LINEAR
            )
        else:
            mask_resized = mask_data

        # Convert to uint8
        mask_uint8 = (mask_resized * 255).astype(np.uint8)

        # Crop to bounding box
        mask_cropped = mask_uint8[y1:y2, x1:x2]

        # Apply feathering if configured
        if self.config.segmentation.feather_edges > 0:
            kernel_size = self.config.segmentation.feather_edges * 2 + 1
            mask_cropped = cv2.GaussianBlur(
                mask_cropped, (kernel_size, kernel_size), 0
            )

        return mask_cropped
