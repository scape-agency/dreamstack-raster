"""Resize to aspect ratio operation."""

# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from ._get_cv2_interpolation import _get_cv2_interpolation

if TYPE_CHECKING:
    from numpy.typing import NDArray

Interpolation = Literal["nearest", "linear", "cubic", "lanczos", "area"]


def resize_to_aspect(
    image: NDArray[np.uint8],
    target_width: int,
    aspect_ratio: tuple[int, int] = (16, 9),
    *,
    divisible_by: int = 1,
    interpolation: Interpolation = "lanczos",
) -> NDArray[np.uint8]:
    """Resize image to target width with specific aspect ratio.

    Useful for preparing images for AI models or video output.

    Args:
        image: Input image.
        target_width: Target width in pixels.
        aspect_ratio: Target aspect ratio as (width, height).
        divisible_by: Ensure dimensions are divisible by this value.
        interpolation: Interpolation method.

    Returns:
        Resized and cropped/padded image.

    Example:
        >>> # Resize to 1920x1080 (16:9 aspect)
        >>> result = resize_to_aspect(image, 1920, (16, 9))
        >>>
        >>> # Resize for AI model (dimensions divisible by 64)
        >>> result = resize_to_aspect(image, 1024, (16, 9), divisible_by=64)
    """
    import cv2

    ar_w, ar_h = aspect_ratio
    target_height = int(target_width * ar_h / ar_w)

    # Adjust for divisibility
    if divisible_by > 1:
        target_width = (target_width // divisible_by) * divisible_by
        target_height = (target_height // divisible_by) * divisible_by

    interp = _get_cv2_interpolation(interpolation)
    return np.asarray(
        cv2.resize(image, (target_width, target_height), interpolation=interp),
        dtype=np.uint8,
    )
