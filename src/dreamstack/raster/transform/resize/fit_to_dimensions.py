"""Fit to dimensions operation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from .resize import resize

if TYPE_CHECKING:
    from numpy.typing import NDArray

Interpolation = Literal["nearest", "linear", "cubic", "lanczos", "area"]


def fit_to_dimensions(
    image: NDArray[np.uint8],
    max_width: int,
    max_height: int,
    *,
    interpolation: Interpolation = "lanczos",
    allow_upscale: bool = False,
) -> NDArray[np.uint8]:
    """Fit image within maximum dimensions, preserving aspect ratio.

    The image will be scaled down (or optionally up) to fit within
    the specified bounds while maintaining its aspect ratio.

    Args:
        image: Input image.
        max_width: Maximum allowed width.
        max_height: Maximum allowed height.
        interpolation: Interpolation method.
        allow_upscale: Allow scaling up smaller images.

    Returns:
        Fitted image.

    Example:
        >>> # Fit within 1920x1080 bounds
        >>> fitted = fit_to_dimensions(image, 1920, 1080)
    """
    h, w = image.shape[:2]

    # Calculate scale factors
    scale_w = max_width / w
    scale_h = max_height / h
    scale = min(scale_w, scale_h)

    # Don't upscale unless allowed
    if not allow_upscale and scale > 1.0:
        return image.copy()

    new_width = int(w * scale)
    new_height = int(h * scale)

    return resize(image, (new_width, new_height), interpolation=interpolation)
