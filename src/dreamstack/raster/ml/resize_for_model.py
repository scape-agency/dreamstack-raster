# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Resize for model operation."""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def resize_for_model(
    image: NDArray[np.uint8],
    target_size: tuple[int, int],
    *,
    preserve_aspect: bool = True,
    pad_color: tuple[int, int, int] = (0, 0, 0),
    interpolation: int = cv2.INTER_LINEAR,
) -> NDArray[np.uint8]:
    """Resize image for model input with optional padding.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input image.
    target_size : tuple[int, int]
        Target (width, height).
    preserve_aspect : bool, optional
        Maintain aspect ratio. Default is True.
    pad_color : tuple, optional
        Padding color. Default is black.
    interpolation : int, optional
        OpenCV interpolation method.

    Returns
    -------
    NDArray[np.uint8]
        Resized image.
    """
    if not preserve_aspect:
        return cv2.resize(image, target_size, interpolation=interpolation)

    h, w = image.shape[:2]
    target_w, target_h = target_size

    # Calculate scale to fit
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    # Create padded output
    if image.ndim == 3:
        result = np.full(
            (target_h, target_w, image.shape[2]), pad_color, dtype=np.uint8
        )
    else:
        result = np.full((target_h, target_w), pad_color[0], dtype=np.uint8)

    # Center the resized image
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    result[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

    return result
