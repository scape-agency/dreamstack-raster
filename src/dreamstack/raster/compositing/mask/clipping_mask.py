# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Clipping Mask
=============

Create clipping masks from layers.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

# =============================================================================
# Type Checking Imports
# =============================================================================

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel
    from numpy.typing import NDArray


def create_clipping_mask(
    content: NDArray[np.uint8],
    clip_to: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    """Create a clipping mask effect.

    The content is clipped to the visible area of the clip_to layer.
    Like Photoshop's clipping mask where one layer clips to another.

    Args:
        content: The layer to be clipped (will show through).
        clip_to: The layer that defines the clipping area.

    Returns:
        Content image clipped to clip_to's alpha.

    Example:
        >>> # Clip a texture to a shape
        >>> result = create_clipping_mask(texture, shape_layer)
    """
    # Ensure both are BGRA
    if content.ndim == 2:
        result = cv2.cvtColor(content, cv2.COLOR_GRAY2BGRA)
    elif content.shape[2] == 3:
        result = cv2.cvtColor(content, cv2.COLOR_BGR2BGRA)
    else:
        result = content.copy()

    # Get clip alpha
    if clip_to.ndim == 2:
        clip_alpha = clip_to
    elif clip_to.shape[2] == 4:
        clip_alpha = clip_to[:, :, 3]
    else:
        # Use luminosity if no alpha
        clip_alpha = cv2.cvtColor(clip_to, cv2.COLOR_BGR2GRAY)

    # Resize if needed
    if clip_alpha.shape[:2] != result.shape[:2]:
        clip_alpha = cv2.resize(clip_alpha, (result.shape[1], result.shape[0]))

    # Apply clip alpha to content
    if result.shape[2] == 4:
        content_alpha = result[:, :, 3].astype(np.float32) / 255.0
        clip_f = clip_alpha.astype(np.float32) / 255.0
        result[:, :, 3] = (content_alpha * clip_f * 255).astype(np.uint8)
    else:
        result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
        result[:, :, 3] = clip_alpha

    return result
