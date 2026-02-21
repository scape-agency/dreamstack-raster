"""Fit operation."""

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


def fit(
    image: NDArray[np.uint8],
    max_size: tuple[int, int],
    *,
    interpolation: Interpolation = "lanczos",
) -> NDArray[np.uint8]:
    """Fit image within maximum dimensions, preserving aspect ratio.

    The image is scaled down to fit entirely within the given dimensions.
    Will not upscale smaller images.

    Args:
        image: Input image.
        max_size: Maximum (width, height).
        interpolation: Interpolation method.

    Returns:
        Fitted image.

    Example:
        >>> fitted = fit(image, (1920, 1080))
    """
    h, w = image.shape[:2]
    max_w, max_h = max_size

    if w <= max_w and h <= max_h:
        return image.copy()

    scale_factor = min(max_w / w, max_h / h)
    new_size = (int(w * scale_factor), int(h * scale_factor))

    return resize(image, new_size, interpolation=interpolation)
