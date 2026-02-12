"""Alpha channel application from mask."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def apply_alpha_from_mask(
    image: NDArray[np.uint8],
    mask: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    """Apply mask as alpha channel to create RGBA image.

    Args:
        image: Input RGB image (3 channels).
        mask: Grayscale mask (white = opaque).

    Returns:
        RGBA image with mask as alpha.

    Example:
        >>> rgba = apply_alpha_from_mask(rgb_image, transparency_mask)
    """
    import cv2

    # Ensure image is 3 channel
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = image[:, :, :3]

    # Ensure mask is single channel
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Resize mask if needed
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    # Create RGBA
    rgba = np.dstack([image, mask])

    return rgba
