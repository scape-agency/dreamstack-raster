# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Base Detector
=============

Abstract base class for object detection backends.
"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    # pylint: disable=import-outside-toplevel
    from numpy.typing import NDArray

    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.detection.config import DetectionConfig

    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.detection.result import ImageDetectionResult


class BaseDetector(ABC):
    """Abstract base class for object detection.

    Subclasses must implement:
    - `_load_model()`: Load the detection model
    - `detect()`: Run detection on an image
    - `get_class_names()`: Return mapping of class IDs to names

    Example
    -------
    >>> class MyDetector(BaseDetector):
    ...     def _load_model(self):
    ...         self.model = load_my_model()
    ...
    ...     def detect(self, image):
    ...         return self.model.predict(image)
    """

    def __init__(self, config: DetectionConfig) -> None:
        """Initialize detector with configuration.

        Parameters
        ----------
        config : DetectionConfig
            Detection configuration.
        """
        self.config = config
        self._model = None
        self._model_loaded = False

    @abstractmethod
    def _load_model(self) -> None:
        """Load the detection model.

        Called lazily on first detection. Subclasses should
        set self._model and self._model_loaded = True.
        """
        ...  # pylint: disable=unnecessary-ellipsis

    @abstractmethod
    def detect(self, image: NDArray[np.uint8]) -> ImageDetectionResult:
        """Detect objects in an image.

        Parameters
        ----------
        image : NDArray[np.uint8]
            Input image (BGR or RGB, HxWxC).

        Returns
        -------
        ImageDetectionResult
            Detection results including bounding boxes, labels, and masks.
        """
        ...  # pylint: disable=unnecessary-ellipsis

    @abstractmethod
    def get_class_names(self) -> dict[int, str]:
        """Get mapping of class IDs to human-readable names.

        Returns
        -------
        dict[int, str]
            Mapping from class ID to class name.
        """
        ...  # pylint: disable=unnecessary-ellipsis

    def _ensure_model_loaded(self) -> None:
        """Ensure model is loaded before detection."""
        if not self._model_loaded:
            self._load_model()
            self._model_loaded = True

    def _resolve_device(self) -> str:
        """Resolve device string to appropriate value for the backend.

        Returns
        -------
        str
            Device identifier ("cpu", "cuda", "mps", or device index).
        """
        if self.config.device == "auto":
            try:
                import torch

                if torch.cuda.is_available():
                    return "cuda"
                elif (
                    hasattr(torch.backends, "mps")
                    and torch.backends.mps.is_available()
                ):
                    return "mps"
                else:
                    return "cpu"
            except ImportError:
                return "cpu"
        return self.config.device
