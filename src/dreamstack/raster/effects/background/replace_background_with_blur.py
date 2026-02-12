"""
Replace Background with Blur
============================

Replace background with blurred version of the original image.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def replace_background_with_blur(
    image: NDArray[np.uint8],
    rgba_or_mask: NDArray[np.uint8],
    blur_radius: int = 21,
) -> NDArray[np.uint8]:
    """Replace background with blurred version of the original image.

    Creates a depth-of-field effect by blurring the background while
    keeping the foreground sharp.

    Args:
        image: Original image (BGR, 3 channels).
        rgba_or_mask: Either RGBA image or grayscale mask.
        blur_radius: Gaussian blur radius (must be odd).

    Returns:
        Image with blurred background.

    Example:
        >>> mask = extract_alpha_mask(image)
        >>> result = replace_background_with_blur(image, mask, blur_radius=31)
    """
    import cv2

    # Ensure blur radius is odd
    if blur_radius % 2 == 0:
        blur_radius += 1

    # Extract mask
    if rgba_or_mask.ndim == 3 and rgba_or_mask.shape[2] == 4:
        mask = rgba_or_mask[:, :, 3]
    elif rgba_or_mask.ndim == 2:
        mask = rgba_or_mask
    else:
        raise ValueError("Expected RGBA image or grayscale mask")

    # Ensure mask matches image size
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(
            mask,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    # Create blurred background
    blurred = cv2.GaussianBlur(image, (blur_radius, blur_radius), 0)

    # Normalize mask for blending
    alpha = mask.astype(np.float32) / 255.0
    alpha = np.expand_dims(alpha, axis=-1)

    # Blend: foreground is sharp, background is blurred
    result = image.astype(np.float32) * alpha + blurred.astype(np.float32) * (1 - alpha)

    return result.astype(np.uint8)
