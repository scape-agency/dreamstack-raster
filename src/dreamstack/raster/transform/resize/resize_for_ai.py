"""Resize for AI/ML models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from .resize_to_aspect import resize_to_aspect

if TYPE_CHECKING:
    from numpy.typing import NDArray

Interpolation = Literal["nearest", "linear", "cubic", "lanczos", "area"]


def resize_for_ai(
    image: NDArray[np.uint8],
    target_width: int = 1024,
    *,
    divisible_by: int = 64,
    aspect_ratio: tuple[int, int] = (16, 9),
    interpolation: Interpolation = "lanczos",
) -> NDArray[np.uint8]:
    """Resize image for AI/ML model input.

    Ensures dimensions are divisible by a power of 2 for neural networks.

    Args:
        image: Input image.
        target_width: Target width.
        divisible_by: Ensure dimensions divisible by this (typically 8, 16, 32, or 64).
        aspect_ratio: Target aspect ratio.
        interpolation: Interpolation method.

    Returns:
        Resized image with AI-compatible dimensions.

    Example:
        >>> # Prepare for Stable Diffusion (768x512, divisible by 8)
        >>> ai_ready = resize_for_ai(image, 768, divisible_by=8, aspect_ratio=(3, 2))
    """
    return resize_to_aspect(
        image,
        target_width,
        aspect_ratio,
        divisible_by=divisible_by,
        interpolation=interpolation,
    )
