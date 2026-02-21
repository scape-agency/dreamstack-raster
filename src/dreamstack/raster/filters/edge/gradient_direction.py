# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Gradient Direction
======================================

Gradient magnitude and direction calculation implementation.

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
    from dreamstack.raster.core.image import Image


def gradient_direction(image: Image) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate gradient magnitude and direction.

    Args:
        image: Input image

    Returns:
        Tuple of (magnitude, direction in radians)
    """
    gray_img = image.to_grayscale() if image.channels > 1 else image
    data = gray_img.data.astype(np.float32)

    if data.ndim == 3:
        data = data[:, :, 0]

    sobelx = cv2.Sobel(data, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(data, cv2.CV_32F, 0, 1, ksize=3)

    magnitude = np.sqrt(sobelx**2 + sobely**2)
    direction = np.arctan2(sobely, sobelx)

    return magnitude, direction
