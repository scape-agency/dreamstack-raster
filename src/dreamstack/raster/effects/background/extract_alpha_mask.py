# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Extract Alpha Mask
==================

Extract alpha mask from images using AI segmentation.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from dreamstack.raster.effects.background.removal_config import (
    ModelName,
    RemovalConfig,
)
from dreamstack.raster.effects.background.remove_background import (
    remove_background,
)

# =============================================================================
# Type Checking Imports
# =============================================================================

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel
    from numpy.typing import NDArray


def extract_alpha_mask(
    image: NDArray[np.uint8],
    config: RemovalConfig | None = None,
    *,
    model_name: ModelName | None = None,
    threshold: int | None = None,
) -> NDArray[np.uint8]:
    """Extract alpha mask from an image without removing background.

    Returns a single-channel grayscale mask where white (255) represents
    foreground and black (0) represents background.

    Args:
        image: Input image as numpy array (BGR, 3 channels).
        config: Optional RemovalConfig for segmentation settings.
        model_name: Override model name.
        threshold: Apply binary threshold to mask (0-255). None for soft mask.

    Returns:
        Grayscale alpha mask (single channel, 0-255).

    Example:
        >>> mask = extract_alpha_mask(image, model_name="u2net")
        >>> binary_mask = extract_alpha_mask(image, threshold=128)
    """
    # Get RGBA result
    rgba = remove_background(image, config, model_name=model_name)

    # Extract alpha channel
    mask = rgba[:, :, 3]

    # Apply threshold if specified
    if threshold is not None:
        mask = np.where(mask >= threshold, 255, 0).astype(np.uint8)

    return mask
