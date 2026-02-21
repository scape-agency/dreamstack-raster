"""Fill operation."""

# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from .resize import resize

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel
    from numpy.typing import NDArray

Interpolation = Literal["nearest", "linear", "cubic", "lanczos", "area"]


def fill(
    image: NDArray[np.uint8],
    target_size: tuple[int, int],
    *,
    interpolation: Interpolation = "lanczos",
) -> NDArray[np.uint8]:
    """Fill target dimensions, cropping if necessary.

    The image is scaled to completely fill the target area,
    then center-cropped to exact dimensions.

    Args:
        image: Input image.
        target_size: Target (width, height).
        interpolation: Interpolation method.

    Returns:
        Filled and cropped image.

    Example:
        >>> filled = fill(image, (1920, 1080))
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size

    # Scale to fill (larger dimension matches, may overflow)
    scale_factor = max(target_w / w, target_h / h)
    scaled_w = int(w * scale_factor)
    scaled_h = int(h * scale_factor)

    scaled = resize(image, (scaled_w, scaled_h), interpolation=interpolation)

    # Center crop
    x_offset = (scaled_w - target_w) // 2
    y_offset = (scaled_h - target_h) // 2

    return scaled[
        y_offset : y_offset + target_h, x_offset : x_offset + target_w
    ]
