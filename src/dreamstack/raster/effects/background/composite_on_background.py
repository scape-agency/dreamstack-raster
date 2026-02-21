# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Composite on Background
=======================

Composite RGBA images onto solid color backgrounds.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel
    from numpy.typing import NDArray


def composite_on_background(
    rgba_image: NDArray[np.uint8],
    background_color: tuple[int, int, int] = (255, 255, 255),
) -> NDArray[np.uint8]:
    """
    Composite an RGBA image onto a solid color background.

    Blend the RGBA image with the alpha channel onto a solid background.

    Args:
        rgba_image: RGBA image with alpha channel (4 channels).
        background_color: RGB tuple for background (default white).

    Returns:
        RGB image composited on background (3 channels).

    Example:
        >>> # Composite on white background
        >>> rgb = composite_on_background(rgba_image)
        >>> # Composite on black background
        >>> rgb = composite_on_background(rgba_image, (0, 0, 0))
    """
    if rgba_image.ndim != 3 or rgba_image.shape[2] != 4:
        raise ValueError("Expected RGBA image with 4 channels")

    # Extract channels
    rgb = rgba_image[:, :, :3].astype(np.float32)
    alpha = rgba_image[:, :, 3:4].astype(np.float32) / 255.0

    # Create background
    bg = np.full_like(rgb, background_color, dtype=np.float32)

    # Alpha blend
    result = rgb * alpha + bg * (1 - alpha)

    return result.astype(np.uint8)
