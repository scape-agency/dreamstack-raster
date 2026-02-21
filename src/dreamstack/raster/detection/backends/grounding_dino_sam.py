"""
Grounding DINO + SAM Backend
============================

Open-vocabulary object detection using Grounding DINO
with precise segmentation from Segment Anything (SAM).

This backend allows text-prompted detection of arbitrary objects.
"""

from __future__ import annotations

import logging
from pathlib import Path
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


class GroundingDinoSamDetector(BaseDetector):
    """Open-vocabulary detection with Grounding DINO + SAM.

    Detects objects based on text prompts and generates
    precise segmentation masks using SAM.

    Example
    -------
    >>> from dreamstack.raster.detection import DetectionConfig
    >>> from dreamstack.raster.detection.backends import GroundingDinoSamDetector
    >>>
    >>> config = DetectionConfig(
    ...     text_prompts=["person", "face", "eye", "earring", "hand"],
    ...     confidence_threshold=0.3,
    ... )
    >>> detector = GroundingDinoSamDetector(config)
    >>> result = detector.detect(image)

    Notes
    -----
    Requires:
        pip install groundingdino-py segment-anything

    Models are downloaded automatically on first use (~2GB total).
    """

    # Model configs
    GROUNDING_DINO_CONFIG = "GroundingDINO_SwinT_OGC.py"
    GROUNDING_DINO_CHECKPOINT = "groundingdino_swint_ogc.pth"
    SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"

    def __init__(self, config: DetectionConfig | None = None) -> None:
        """Initialize detector.

        Parameters
        ----------
        config : DetectionConfig | None
            Detection configuration.
        """
        super().__init__(config or DetectionConfig())

        # Text prompts for detection
        self._text_prompts: list[str] = getattr(
            self.config, "text_prompts", ["person", "object"]
        )

        self._grounding_dino_model = None
        self._sam_predictor = None
        self._effective_device = "cpu"  # Set properly in _load_model

    def set_prompts(self, prompts: list[str] | str) -> None:
        """Set text prompts for detection.

        Parameters
        ----------
        prompts : list[str] | str
            Object names to detect (e.g., ["person", "face", "eye"]).
            Can also be comma-separated string.
        """
        if isinstance(prompts, str):
            self._text_prompts = [
                p.strip() for p in prompts.split(",") if p.strip()
            ]
        else:
            self._text_prompts = list(prompts)

    def get_prompts(self) -> list[str]:
        """Get current text prompts."""
        return self._text_prompts.copy()

    def _load_model(self) -> None:
        """Load Grounding DINO and SAM models."""
        # Determine device once - force CPU if MPS (CUDA-only code in libs)
        device = self._resolve_device()
        if device == "mps":
            self._effective_device = "cpu"
            logger.info("Using CPU for models (MPS has CUDA-only lib issues)")
        else:
            self._effective_device = device

        self._load_grounding_dino()
        self._load_sam()
        self._model_loaded = True

    def _load_grounding_dino(self) -> None:
        """Load Grounding DINO model."""
        try:
            # pylint: disable=import-outside-toplevel
            from groundingdino.util.inference import (
                load_model,
            )  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "groundingdino-py is required for Grounding DINO. "
                "Install with: pip install groundingdino-py"
            ) from exc

        # Download model if needed
        model_dir = self._get_model_dir()
        config_path = model_dir / self.GROUNDING_DINO_CONFIG
        checkpoint_path = model_dir / self.GROUNDING_DINO_CHECKPOINT

        if not checkpoint_path.exists():
            self._download_grounding_dino_model(model_dir)

        logger.info("Loading Grounding DINO on %s", self._effective_device)

        self._grounding_dino_model = load_model(
            str(config_path),
            str(checkpoint_path),
            device=self._effective_device,
        )

    def _load_sam(self) -> None:
        """Load SAM model."""
        try:
            # pylint: disable=import-outside-toplevel
            from segment_anything import (  # type: ignore[import-not-found]
                SamPredictor,
                sam_model_registry,
            )
        except ImportError as exc:
            raise ImportError(
                "segment-anything is required for SAM. "
                "Install with: pip install segment-anything"
            ) from exc

        model_dir = self._get_model_dir()
        checkpoint_path = model_dir / self.SAM_CHECKPOINT

        if not checkpoint_path.exists():
            self._download_sam_model(model_dir)

        logger.info("Loading SAM on %s", self._effective_device)

        sam = sam_model_registry["vit_h"](checkpoint=str(checkpoint_path))
        sam.to(self._effective_device)

        self._sam_predictor = SamPredictor(sam)

    def _get_model_dir(self) -> Path:
        """Get model cache directory."""
        cache_dir = Path.home() / ".cache" / "dreamstack" / "models"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _download_grounding_dino_model(self, model_dir: Path) -> None:
        """Download Grounding DINO model files."""
        import urllib.request  # pylint: disable=import-outside-toplevel

        logger.info("Downloading Grounding DINO model...")

        config_url = (
            "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/"
            "main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        )
        checkpoint_url = (
            "https://github.com/IDEA-Research/GroundingDINO/releases/download/"
            "v0.1.0-alpha/groundingdino_swint_ogc.pth"
        )

        urllib.request.urlretrieve(
            config_url,
            model_dir / self.GROUNDING_DINO_CONFIG,
        )
        urllib.request.urlretrieve(
            checkpoint_url,
            model_dir / self.GROUNDING_DINO_CHECKPOINT,
        )

    def _download_sam_model(self, model_dir: Path) -> None:
        """Download SAM model checkpoint."""
        import urllib.request  # pylint: disable=import-outside-toplevel

        logger.info("Downloading SAM model (this may take a while, ~2.4GB)...")

        sam_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

        urllib.request.urlretrieve(
            sam_url,
            model_dir / self.SAM_CHECKPOINT,
        )

    def get_class_names(self) -> dict[int, str]:
        """Get mapping of prompt index to prompt text.

        Returns
        -------
        dict[int, str]
            Mapping from index to prompt.
        """
        return {i: prompt for i, prompt in enumerate(self._text_prompts)}

    def detect(
        self,
        image: NDArray[np.uint8],
        prompts: list[str] | str | None = None,
    ) -> ImageDetectionResult:
        """Detect objects matching text prompts.

        Parameters
        ----------
        image : NDArray[np.uint8]
            Input image (BGR format, HxWxC).
        prompts : list[str] | str | None
            Optional prompts override. Uses configured prompts if None.

        Returns
        -------
        ImageDetectionResult
            Detection results with boxes, labels, and masks.
        """
        self._ensure_model_loaded()

        if prompts is not None:
            self.set_prompts(prompts)

        h, w = image.shape[:2]

        # Convert BGR to RGB for models
        import cv2  # pylint: disable=import-outside-toplevel

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Run Grounding DINO
        boxes, labels, confidences = self._run_grounding_dino(image_rgb)

        # Generate masks with SAM
        masks = self._run_sam(image_rgb, boxes) if len(boxes) > 0 else []

        # Build results
        detections: list[DetectionResult] = []

        for i, (box, label, conf) in enumerate(
            zip(boxes, labels, confidences)
        ):
            if conf < self.config.confidence_threshold:
                continue

            x1, y1, x2, y2 = map(int, box)
            bbox_width = x2 - x1
            bbox_height = y2 - y1

            # Skip tiny detections
            if (
                bbox_width < self.config.min_dimension
                or bbox_height < self.config.min_dimension
            ):
                continue

            # Get mask for this detection
            mask = None
            if i < len(masks) and self.config.segmentation.enabled:
                mask = self._crop_mask(masks[i], (x1, y1, x2, y2))

            detections.append(
                DetectionResult(
                    label=label,
                    class_id=(
                        self._text_prompts.index(label)
                        if label in self._text_prompts
                        else i
                    ),
                    confidence=float(conf),
                    bbox=(x1, y1, bbox_width, bbox_height),
                    mask=mask,
                )
            )

        return ImageDetectionResult(
            source_path=None,
            image_size=(h, w),
            detections=detections,
        )

    def _run_grounding_dino(
        self,
        image_rgb: NDArray[np.uint8],
    ) -> tuple[list, list[str], list[float]]:
        """Run Grounding DINO detection.

        Parameters
        ----------
        image_rgb : NDArray[np.uint8]
            RGB image.

        Returns
        -------
        tuple[list, list[str], list[float]]
            (boxes, labels, confidences)
        """
        # pylint: disable=import-outside-toplevel
        from groundingdino.util.inference import predict

        # Combine prompts with periods (Grounding DINO format)
        text_prompt = ". ".join(self._text_prompts) + "."

        boxes, logits, phrases = predict(
            model=self._grounding_dino_model,
            image=image_rgb,
            caption=text_prompt,
            box_threshold=self.config.confidence_threshold,
            text_threshold=self.config.confidence_threshold,
            device=self._effective_device,
        )

        # Convert boxes from normalized to pixel coordinates
        h, w = image_rgb.shape[:2]
        boxes_pixel = []
        for box in boxes:
            cx, cy, bw, bh = box
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            boxes_pixel.append([x1, y1, x2, y2])

        return boxes_pixel, list(phrases), logits.tolist()

    def _run_sam(
        self,
        image_rgb: NDArray[np.uint8],
        boxes: list,
    ) -> list[NDArray[np.uint8]]:
        """Run SAM to generate masks for detected boxes.

        Parameters
        ----------
        image_rgb : NDArray[np.uint8]
            RGB image.
        boxes : list
            List of bounding boxes [x1, y1, x2, y2].

        Returns
        -------
        list[NDArray[np.uint8]]
            List of masks (full image size).
        """
        self._sam_predictor.set_image(image_rgb)  # type: ignore[union-attr]

        masks = []
        for box in boxes:
            # Use numpy-based predict (works on MPS/CPU)
            # box format: [x1, y1, x2, y2]
            box_array = np.array(box)

            mask_output, _, _ = self._sam_predictor.predict(  # type: ignore[union-attr]
                point_coords=None,
                point_labels=None,
                box=box_array,
                multimask_output=False,
            )

            # Convert to uint8 mask (mask_output shape: [1, H, W])
            mask = (mask_output[0] * 255).astype(np.uint8)  # type: ignore[union-attr]
            masks.append(mask)

        return masks

    def _crop_mask(
        self,
        mask: NDArray[np.uint8],
        bbox: tuple[int, int, int, int],
    ) -> NDArray[np.uint8]:
        """Crop mask to bounding box region.

        Parameters
        ----------
        mask : NDArray[np.uint8]
            Full-size mask.
        bbox : tuple[int, int, int, int]
            Bounding box (x1, y1, x2, y2).

        Returns
        -------
        NDArray[np.uint8]
            Cropped mask.
        """
        x1, y1, x2, y2 = bbox
        return mask[y1:y2, x1:x2].copy()
