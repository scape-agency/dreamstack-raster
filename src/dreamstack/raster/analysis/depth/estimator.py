# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Depth Estimator Class
=====================

Class-based interface for monocular depth estimation
using transformer-based models like Depth Anything.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

# =============================================================================
# Type Checking Imports
# =============================================================================

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel
    from numpy.typing import NDArray


# Available depth models
ModelSize = Literal["small", "base", "large"]
ModelName = Literal[
    "LiheYoung/depth-anything-small-hf",
    "LiheYoung/depth-anything-base-hf",
    "LiheYoung/depth-anything-large-hf",
    "Intel/dpt-hybrid-midas",
    "Intel/dpt-large",
]


@dataclass
class DepthConfig:
    """
    Configuration for depth estimation.

    Attributes:
        model_name: HuggingFace model identifier.
        device: Device to run inference on (auto, cpu, cuda, mps).
        normalize_output: Whether to normalize depth to 0-1 range.
        invert: Invert depth map (closer = brighter).
    """

    model_name: ModelName = "LiheYoung/depth-anything-base-hf"
    device: str = "auto"
    normalize_output: bool = True
    invert: bool = False


@dataclass
class DepthResult:
    """
    Result from depth estimation.

    Attributes:
        depth_map: Raw depth values as float32 array.
        depth_normalized: Normalized depth map (0-1 range).
        min_depth: Minimum depth value.
        max_depth: Maximum depth value.
        model_name: Model used for estimation.
    """

    depth_map: NDArray[np.float32]
    depth_normalized: NDArray[np.float32]
    min_depth: float
    max_depth: float
    model_name: str

    def to_uint8(self) -> NDArray[np.uint8]:
        """Convert normalized depth to uint8 image."""
        return (self.depth_normalized * 255).astype(np.uint8)

    def to_colormap(self, colormap: str = "inferno") -> NDArray[np.uint8]:
        """Convert depth to colored visualization.

        Args:
            colormap: Matplotlib colormap name.

        Returns:
            RGB image with depth colorized.
        """
        try:
            import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise ImportError(
                "matplotlib required for colormaps. "
                "Install with: pip install matplotlib"
            ) from exc

        cmap = plt.get_cmap(colormap)
        colored = cmap(self.depth_normalized)[:, :, :3]
        return (colored * 255).astype(np.uint8)


class DepthEstimator:
    """AI-based monocular depth estimation.

    Uses transformer models (Depth Anything, DPT) for estimating
    depth from single images.

    Example:
        >>> from dreamstack.raster.analysis.depth import DepthEstimator
        >>> estimator = DepthEstimator()
        >>> result = estimator.estimate(image)
        >>> depth_image = result.to_uint8()

    Example with custom model:
        >>> config = DepthConfig(
        ...     model_name="LiheYoung/depth-anything-large-hf"
        ... )
        >>> estimator = DepthEstimator(config)
    """

    def __init__(self, config: DepthConfig | None = None):
        """Initialize depth estimator.

        Args:
            config: Optional configuration.
        """
        self.config = config or DepthConfig()
        self._pipe = None
        self._model_loaded = False

    def _load_model(self) -> None:
        """Lazy load the depth estimation model."""
        if self._model_loaded:
            return

        try:
            # pylint: disable=import-outside-toplevel
            from transformers import pipeline
        except ImportError as exc:
            raise ImportError(
                "transformers is required for depth estimation. "
                "Install with: pip install transformers torch"
            ) from exc

        device = self._resolve_device()

        self._pipe = pipeline(
            "depth-estimation",
            model=self.config.model_name,
            device=device,
        )
        self._model_loaded = True

    def _resolve_device(self) -> int | str:
        """Resolve device string to appropriate value."""
        if self.config.device == "auto":
            try:
                import torch  # pylint: disable=import-outside-toplevel

                if torch.cuda.is_available():
                    return 0  # GPU
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

    def estimate(self, image: NDArray[np.uint8]) -> DepthResult:
        """Estimate depth from a single image.

        Args:
            image: Input image (BGR or RGB, 3 channels).

        Returns:
            DepthResult with depth map and metadata.
        """
        # pylint: disable=import-outside-toplevel
        import cv2

        # pylint: disable=import-outside-toplevel
        from PIL import Image

        self._load_model()

        # Convert to PIL
        if image.ndim == 3 and image.shape[2] == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb = image

        pil_image = Image.fromarray(rgb)

        # Run inference
        if self._pipe is None:
            raise RuntimeError("Depth estimation pipeline not initialized")
        result = self._pipe(pil_image)

        # Extract depth map
        depth_map = np.array(result["depth"]).astype(np.float32)  # type: ignore[index]

        # Normalize
        min_depth = float(depth_map.min())
        max_depth = float(depth_map.max())

        if max_depth > min_depth:
            depth_normalized = (depth_map - min_depth) / (
                max_depth - min_depth
            )
        else:
            depth_normalized = np.zeros_like(depth_map)

        # Invert if configured
        if self.config.invert:
            depth_normalized = 1.0 - depth_normalized

        return DepthResult(
            depth_map=depth_map,
            depth_normalized=depth_normalized.astype(np.float32),
            min_depth=min_depth,
            max_depth=max_depth,
            model_name=self.config.model_name,
        )

    def estimate_batch(
        self,
        images: list[NDArray[np.uint8]],
    ) -> list[DepthResult]:
        """Estimate depth for multiple images.

        Args:
            images: List of input images.

        Returns:
            List of DepthResult objects.
        """
        return [self.estimate(img) for img in images]

    @classmethod
    def with_config(
        cls,
        model_name: ModelName | None = None,
        model_size: ModelSize | None = None,
        invert: bool = False,
    ) -> DepthEstimator:
        """Create estimator with specific configuration.

        Args:
            model_name: Full model name from HuggingFace.
            model_size: Shortcut for Depth Anything model size.
            invert: Invert depth map.

        Returns:
            Configured DepthEstimator instance.

        Example:
            >>> estimator = DepthEstimator.with_config(model_size="large")
        """
        if model_size is not None:
            size_map: dict[str, ModelName] = {
                "small": "LiheYoung/depth-anything-small-hf",
                "base": "LiheYoung/depth-anything-base-hf",
                "large": "LiheYoung/depth-anything-large-hf",
            }
            model_name = size_map.get(model_size, size_map["base"])

        config = DepthConfig(
            model_name=model_name or "LiheYoung/depth-anything-base-hf",
            invert=invert,
        )
        return cls(config)
