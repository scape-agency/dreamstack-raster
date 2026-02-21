# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Apply Mask
==========

Apply a mask to an image.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def apply_mask(
    image: NDArray[np.uint8],
    mask: NDArray[np.uint8],
    *,
    invert: bool = False,
) -> NDArray[np.uint8]:
    """Apply a mask to an image.

    The mask controls transparency: white = opaque, black = transparent.

    Args:
        image: Input image (BGR or BGRA).
        mask: Grayscale mask image.
        invert: If True, invert the mask before applying.

    Returns:
        Image with mask applied as alpha channel (BGRA).

    Example:
        >>> masked = apply_mask(image, gradient_mask)
        >>> inverted = apply_mask(image, mask, invert=True)
    """
    # Ensure BGRA output
    if image.ndim == 2:
        result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)  # type: ignore[assignment]
    elif image.shape[2] == 3:
        result = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)  # type: ignore[assignment]
    else:
        result = image.copy()

    # Ensure mask matches image size
    if mask.shape[:2] != result.shape[:2]:
        mask = cv2.resize(mask, (result.shape[1], result.shape[0]))  # type: ignore[assignment]

    # Ensure mask is grayscale
    if mask.ndim > 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)  # type: ignore[assignment]

    # Invert if requested
    if invert:
        mask = 255 - mask

    # Apply mask to existing alpha
    if image.ndim > 2 and image.shape[2] == 4:
        # Combine with existing alpha
        existing_alpha = result[:, :, 3].astype(np.float32) / 255.0
        mask_alpha = mask.astype(np.float32) / 255.0
        result[:, :, 3] = (existing_alpha * mask_alpha * 255).astype(np.uint8)
    else:
        result[:, :, 3] = mask

    return result
