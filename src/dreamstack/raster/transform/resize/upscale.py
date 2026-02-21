"""Upscale operation."""

# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from .resize import resize

if TYPE_CHECKING:
    from numpy.typing import NDArray

Interpolation = Literal["nearest", "linear", "cubic", "lanczos", "area"]


def upscale(
    image: NDArray[np.uint8],
    scale: float = 2.0,
    *,
    interpolation: Interpolation = "lanczos",
) -> NDArray[np.uint8]:
    """Upscale image by a factor.

    For AI-based upscaling, see the upscale module.

    Args:
        image: Input image.
        scale: Scale factor (> 1.0 for upscaling).
        interpolation: Interpolation method.

    Returns:
        Upscaled image.

    Example:
        >>> upscaled = upscale(image, scale=2.0)
    """
    h, w = image.shape[:2]
    new_width = int(w * scale)
    new_height = int(h * scale)

    return resize(image, (new_width, new_height), interpolation=interpolation)
