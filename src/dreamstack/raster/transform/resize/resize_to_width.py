"""Resize to width operation."""

# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from .resize import resize

# =============================================================================
# Type Checking Imports
# =============================================================================

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel
    from numpy.typing import NDArray

Interpolation = Literal["nearest", "linear", "cubic", "lanczos", "area"]


def resize_to_width(
    image: NDArray[np.uint8],
    target_width: int,
    *,
    interpolation: Interpolation = "lanczos",
) -> NDArray[np.uint8]:
    """Resize image to target width, preserving aspect ratio.

    Args:
        image: Input image.
        target_width: Target width in pixels.
        interpolation: Interpolation method.

    Returns:
        Resized image.

    Example:
        >>> resized = resize_to_width(image, 1920)
    """
    h, w = image.shape[:2]
    scale = target_width / w
    target_height = int(h * scale)

    return resize(
        image, (target_width, target_height), interpolation=interpolation
    )
