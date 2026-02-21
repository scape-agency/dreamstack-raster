"""Scale operation."""

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


def scale(
    image: NDArray[np.uint8],
    factor: float,
    *,
    interpolation: Interpolation = "lanczos",
) -> NDArray[np.uint8]:
    """Scale image by a factor.

    Args:
        image: Input image.
        factor: Scale factor (e.g., 0.5 for half size, 2.0 for double).
        interpolation: Interpolation method.

    Returns:
        Scaled image.

    Example:
        >>> half = scale(image, 0.5)
        >>> double = scale(image, 2.0)
    """
    h, w = image.shape[:2]
    new_size = (int(w * factor), int(h * factor))
    return resize(image, new_size, interpolation=interpolation)
