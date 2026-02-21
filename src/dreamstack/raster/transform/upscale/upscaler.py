"""
Image Upscaler
==============

AI-based image upscaling using PyTorch models.

"""

# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

# =============================================================================
# Type Checking Imports
# =============================================================================

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel
    from numpy.typing import NDArray


@dataclass
class UpscaleConfig:
    """Configuration for image upscaling.

    Attributes:
        scale_factor: Target upscaling factor.
        model_path: Path to pretrained model weights.
        device: Device to run on (auto, cpu, cuda, mps).
        tile_size: Tile size for processing large images.
        tile_overlap: Overlap between tiles.
    """

    scale_factor: int = 2
    model_path: str | Path | None = None
    device: str = "auto"
    tile_size: int = 512
    tile_overlap: int = 32


class BaseUpscaler(ABC):
    """Abstract base class for image upscalers.

    Defines the interface for upscaling implementations.
    """

    @abstractmethod
    def load_model(self, model_path: str | Path) -> None:
        """Load a pretrained upscaling model."""

    @abstractmethod
    def upscale(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Upscale an image."""

    @abstractmethod
    def preprocess(self, image: NDArray[np.uint8]) -> object:
        """Preprocess image for model input."""

    @abstractmethod
    def postprocess(self, output: object) -> NDArray[np.uint8]:
        """Convert model output to image."""


class ImageUpscaler(BaseUpscaler):
    """PyTorch-based image upscaler.

    Uses pretrained super-resolution models to upscale images.
    Supports tile-based processing for large images.

    Example:
        >>> from dreamstack.raster.transform.upscale import ImageUpscaler
        >>> upscaler = ImageUpscaler()
        >>> upscaler.load_model("path/to/model.pth")
        >>> upscaled = upscaler.upscale(image)

    Example with config:
        >>> config = UpscaleConfig(scale_factor=4, device="cuda")
        >>> upscaler = ImageUpscaler(config)
    """

    def __init__(self, config: UpscaleConfig | None = None):
        """Initialize the upscaler.

        Args:
            config: Optional configuration.
        """
        self.config = config or UpscaleConfig()
        self.model = None
        self.device = None
        self._setup_device()

    def _setup_device(self) -> None:
        """Configure the compute device."""
        try:
            import torch  # pylint: disable=import-outside-toplevel
        except ImportError:
            self.device = "cpu"
            return

        if self.config.device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(self.config.device)

    def load_model(self, model_path: str | Path) -> None:
        """Load a pretrained upscaling model.

        Args:
            model_path: Path to model weights (.pth file).

        Raises:
            ImportError: If PyTorch is not installed.
            FileNotFoundError: If model file doesn't exist.
        """
        try:
            import torch  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required for AI upscaling. Install with: pip install torch"
            ) from exc

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model = torch.load(model_path, map_location=self.device)
        self.model.to(self.device)
        self.model.eval()

    def upscale(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Upscale an image using the loaded model.

        Args:
            image: Input image (BGR, 3 channels).

        Returns:
            Upscaled image.

        Raises:
            RuntimeError: If model is not loaded.
        """
        import torch  # pylint: disable=import-outside-toplevel

        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Preprocess
        tensor = self.preprocess(image)

        # Upscale
        with torch.no_grad():
            output = self.model(tensor.to(self.device))  # type: ignore[union-attr]

        # Postprocess
        return self.postprocess(output)

    def preprocess(self, image: NDArray[np.uint8]) -> object:
        """Convert image to model input tensor.

        Args:
            image: Input BGR image.

        Returns:
            PyTorch tensor ready for model.
        """
        import cv2  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel

        # Convert BGR to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to tensor [C, H, W] and normalize to [0, 1]
        tensor = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0

        # Add batch dimension [1, C, H, W]
        return tensor.unsqueeze(0)

    def postprocess(self, output: object) -> NDArray[np.uint8]:
        """Convert model output tensor to image.

        Args:
            output: Model output tensor.

        Returns:
            BGR image as numpy array.
        """
        import cv2  # pylint: disable=import-outside-toplevel

        # Remove batch dimension and convert to numpy
        result = output.squeeze(0).permute(1, 2, 0).cpu().numpy()  # type: ignore[union-attr]

        # Scale back to [0, 255] and clip
        result = (result * 255).clip(0, 255).astype(np.uint8)

        # Convert RGB to BGR
        return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    def upscale_tiled(
        self,
        image: NDArray[np.uint8],
        tile_size: int | None = None,
        overlap: int | None = None,
    ) -> NDArray[np.uint8]:
        """Upscale large image using tiles.

        Processes image in tiles to handle memory constraints.

        Args:
            image: Input image.
            tile_size: Size of each tile (default from config).
            overlap: Overlap between tiles (default from config).

        Returns:
            Upscaled image.
        """

        tile_size = tile_size or self.config.tile_size
        overlap = overlap or self.config.tile_overlap
        scale = self.config.scale_factor

        h, w = image.shape[:2]

        # If image is small enough, process directly
        if h <= tile_size and w <= tile_size:
            return self.upscale(image)

        # Calculate output size
        out_h = h * scale
        out_w = w * scale

        # Create output image
        channels = image.shape[2] if image.ndim == 3 else 1
        output = np.zeros((out_h, out_w, channels), dtype=np.uint8)
        count = np.zeros((out_h, out_w, 1), dtype=np.float32)

        # Process tiles
        step = tile_size - overlap

        for y in range(0, h, step):
            for x in range(0, w, step):
                # Extract tile
                y1 = y
                y2 = min(y + tile_size, h)
                x1 = x
                x2 = min(x + tile_size, w)

                tile = image[y1:y2, x1:x2]

                # Upscale tile
                upscaled_tile = self.upscale(tile)

                # Calculate output positions
                out_y1 = y1 * scale
                out_y2 = y2 * scale
                out_x1 = x1 * scale
                out_x2 = x2 * scale

                # Accumulate
                output[out_y1:out_y2, out_x1:out_x2] += upscaled_tile
                count[out_y1:out_y2, out_x1:out_x2] += 1

        # Average overlapping regions
        count = np.maximum(count, 1)
        output = (output.astype(np.float32) / count).astype(np.uint8)

        return output
