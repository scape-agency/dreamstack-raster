"""Thumbnail generation."""

# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from .fit_to_dimensions import fit_to_dimensions

if TYPE_CHECKING:
    from numpy.typing import NDArray

Interpolation = Literal["nearest", "linear", "cubic", "lanczos", "area"]


def thumbnail(
    image: NDArray[np.uint8],
    max_size: int = 256,
    *,
    interpolation: Interpolation = "area",
) -> NDArray[np.uint8]:
    """Create thumbnail with maximum dimension constraint.

    Args:
        image: Input image.
        max_size: Maximum width or height.
        interpolation: Interpolation method (area is best for downscaling).

    Returns:
        Thumbnail image.

    Example:
        >>> thumb = thumbnail(image, max_size=128)
    """
    return fit_to_dimensions(
        image,
        max_size,
        max_size,
        interpolation=interpolation,
        allow_upscale=False,
    )
