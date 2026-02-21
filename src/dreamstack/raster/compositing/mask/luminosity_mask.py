# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Luminosity Mask
===============

Create masks based on image luminosity.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import cv2
import numpy as np

# =============================================================================
# Type Checking Imports
# =============================================================================

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel
    from numpy.typing import NDArray


def luminosity_mask(
    image: NDArray[np.uint8],
    target: Literal["lights", "darks", "midtones"] = "lights",
    *,
    range_level: int = 1,
    feather: float = 0,
) -> NDArray[np.uint8]:
    """Create a luminosity mask from an image.

    Luminosity masks select pixels based on their brightness
    for precise tonal adjustments.

    Args:
        image: Input image.
        target: Which tones to select ("lights", "darks", "midtones").
        range_level: Refinement level (1-5). Higher = narrower selection.
        feather: Feather amount for smooth edges.

    Returns:
        Grayscale luminosity mask.

    Example:
        >>> # Select highlights for dodge
        >>> highlights = luminosity_mask(image, "lights")
        >>> # Select deep shadows
        >>> shadows = luminosity_mask(image, "darks", range_level=3)
    """
    # Convert to grayscale (luminosity)
    if image.ndim == 2:
        luminosity = image.astype(np.float32) / 255.0
    else:
        # Use ITU-R BT.601 luminosity
        rgb = image[:, :, :3].astype(np.float32) / 255.0
        luminosity = (
            0.299 * rgb[:, :, 2] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 0]
        )

    # Create base mask
    if target == "lights":
        mask = luminosity
    elif target == "darks":
        mask = 1.0 - luminosity
    else:  # midtones
        mask = 1.0 - np.abs(luminosity - 0.5) * 2

    # Apply range level (intersect mask with itself)
    for _ in range(range_level - 1):
        mask = mask * mask

    # Normalize
    if mask.max() > 0:
        mask = mask / mask.max()

    # Feather
    if feather > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), feather)

    return (mask * 255).astype(np.uint8)
