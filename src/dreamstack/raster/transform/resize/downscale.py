"""Downscale operation."""

# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from .fit_to_dimensions import fit_to_dimensions
from .resize import resize

if TYPE_CHECKING:
    from numpy.typing import NDArray

Interpolation = Literal["nearest", "linear", "cubic", "lanczos", "area"]


def downscale(
    image: NDArray[np.uint8],
    max_size: int,
    *,
    preserve_aspect: bool = True,
    interpolation: Interpolation = "area",
) -> NDArray[np.uint8]:
    """Downscale image to fit within maximum size.

    Only scales down, never up. Uses INTER_AREA for best quality.

    Args:
        image: Input image.
        max_size: Maximum width or height.
        preserve_aspect: Keep original aspect ratio.
        interpolation: Interpolation method.

    Returns:
        Downscaled image (or original if already smaller).

    Example:
        >>> smaller = downscale(image, max_size=1024)
    """
    h, w = image.shape[:2]

    if max(w, h) <= max_size:
        return image.copy()

    if preserve_aspect:
        return fit_to_dimensions(
            image,
            max_size,
            max_size,
            interpolation=interpolation,
            allow_upscale=False,
        )
    else:
        return resize(image, (max_size, max_size), interpolation=interpolation)
