"""
Select Focus
============

Select in-focus areas of an image.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from dreamstack.raster.selection.shapes.selection import Selection


def select_focus(
    image: NDArray[np.uint8],
    *,
    threshold: float = 0.5,
    blur_detection_size: int = 3,
) -> Selection:
    """Select in-focus regions of an image.

    Identifies sharp, in-focus areas based on local contrast
    and edge detection.

    Args:
        image: Input image (BGR or BGRA).
        threshold: Focus threshold (0.0-1.0, higher = stricter).
        blur_detection_size: Kernel size for blur detection.

    Returns:
        Selection of in-focus areas.

    Example:
        >>> # Select only sharp areas
        >>> sel = select_focus(image, threshold=0.6)
        >>> sharp_only = sel.apply_to_image(image)
    """
    # Convert to grayscale
    if image.ndim == 2:
        gray = image
    elif image.shape[2] >= 3:
        gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:
        gray = image[:, :, 0]

    h, w = gray.shape[:2]

    # Compute Laplacian for sharpness detection
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=blur_detection_size)
    sharpness = np.abs(laplacian)

    # Normalize to 0-255
    sharpness_norm = cv2.normalize(sharpness, None, 0, 255, cv2.NORM_MINMAX)
    sharpness_u8 = sharpness_norm.astype(np.uint8)

    # Apply local variance as additional measure
    local_mean = cv2.blur(gray.astype(np.float32), (15, 15))
    local_sq_mean = cv2.blur((gray.astype(np.float32) ** 2), (15, 15))
    local_variance = local_sq_mean - (local_mean**2)
    local_variance = np.maximum(0, local_variance)  # Clamp negative values

    # Normalize variance
    variance_norm = cv2.normalize(
        np.sqrt(local_variance), None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    # Combine sharpness and variance
    focus_map = cv2.addWeighted(sharpness_u8, 0.7, variance_norm, 0.3, 0)

    # Apply threshold
    thresh_value = int(threshold * 128)
    _, mask = cv2.threshold(focus_map, thresh_value, 255, cv2.THRESH_BINARY)

    # Clean up with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Smooth edges
    mask = cv2.GaussianBlur(mask, (7, 7), 2)

    return Selection(mask=mask)
