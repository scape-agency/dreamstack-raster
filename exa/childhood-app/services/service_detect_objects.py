"""
Detect Objects Service
======================

Detect objects in images using detection backends.
"""

from __future__ import annotations

import numpy as np


def detect_objects(
    image: np.ndarray,
    backend: str = "ultralytics",
    confidence: float = 0.5,
    prompts: list[str] | None = None,
) -> list[dict]:
    """Detect objects in image.

    Parameters
    ----------
    image : np.ndarray
        Input image (BGR format from OpenCV).
    backend : str
        Detection backend to use. Default "ultralytics".
    confidence : float
        Confidence threshold. Default 0.5.
    prompts : list[str] | None
        Optional text prompts for detection.

    Returns
    -------
    list[dict]
        List of detections with bbox and label.
    """
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
