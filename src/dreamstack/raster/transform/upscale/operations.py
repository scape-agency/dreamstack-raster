# -*- coding: utf-8 -*-

"""
Upscale Operations
==================

Functional interface for image upscaling.

"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


Interpolation = Literal["nearest", "linear", "cubic", "lanczos", "area"]


def upscale_lanczos(
    image: NDArray[np.uint8],
    scale: float = 2.0,
) -> NDArray[np.uint8]:
    """Upscale image using Lanczos interpolation.
    
    High-quality interpolation-based upscaling.
    For AI-based upscaling, use ImageUpscaler class.
    
    Args:
        image: Input image.
        scale: Scale factor (> 1.0).
    
    Returns:
        Upscaled image.
    
    Example:
        >>> upscaled = upscale_lanczos(image, scale=2.0)
    """
    import cv2
    
    h, w = image.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)


def upscale_image(
    image: NDArray[np.uint8],
    scale: float = 2.0,
    *,
    model_path: str | Path | None = None,
    method: Literal["lanczos", "cubic", "model"] = "lanczos",
    device: str = "auto",
) -> NDArray[np.uint8]:
    """Upscale image with specified method.
    
    Unified function for different upscaling approaches.
    
    Args:
        image: Input image.
        scale: Scale factor.
        model_path: Path to model weights (for method="model").
        method: Upscaling method.
        device: Device for model inference.
    
    Returns:
        Upscaled image.
    
    Example:
        >>> # Simple Lanczos upscaling
        >>> result = upscale_image(image, scale=2.0)
        >>> 
        >>> # AI model upscaling
        >>> result = upscale_image(image, scale=4.0, model_path="esrgan.pth", method="model")
    """
    import cv2
    
    if method == "model" and model_path is not None:
        from dreamstack.raster.transform.upscale.upscaler import (
            ImageUpscaler,
            UpscaleConfig,
        )
        
        config = UpscaleConfig(
            scale_factor=int(scale),
            device=device,
        )
        upscaler = ImageUpscaler(config)
        upscaler.load_model(model_path)
        return upscaler.upscale(image)
    
    h, w = image.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    if method == "cubic":
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    else:
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)


def upscale_to_size(
    image: NDArray[np.uint8],
    target_width: int,
    target_height: int | None = None,
    *,
    preserve_aspect: bool = True,
    interpolation: Interpolation = "lanczos",
) -> NDArray[np.uint8]:
    """Upscale image to specific dimensions.
    
    Args:
        image: Input image.
        target_width: Target width.
        target_height: Target height (computed from aspect if None).
        preserve_aspect: Maintain aspect ratio.
        interpolation: Interpolation method.
    
    Returns:
        Upscaled image.
    """
    import cv2
    
    h, w = image.shape[:2]
    
    if target_height is None or preserve_aspect:
        scale = target_width / w
        target_height = int(h * scale)
    
    interp_map = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
        "area": cv2.INTER_AREA,
    }
    
    interp = interp_map.get(interpolation, cv2.INTER_LANCZOS4)
    return cv2.resize(image, (target_width, target_height), interpolation=interp)


def upscale_2x(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Upscale image 2x using Lanczos.
    
    Convenience function for 2x upscaling.
    """
    return upscale_lanczos(image, scale=2.0)


def upscale_4x(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Upscale image 4x using Lanczos.
    
    Convenience function for 4x upscaling.
    """
    return upscale_lanczos(image, scale=4.0)
