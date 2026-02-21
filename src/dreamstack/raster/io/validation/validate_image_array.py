# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Image Array Validation
======================

Validate an image numpy array.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from typing import TYPE_CHECKING

# =============================================================================
# Type Checking Imports
# =============================================================================

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel
    from numpy.typing import NDArray


def validate_image_array(
    image: NDArray,
    *,
    min_channels: int = 1,
    max_channels: int = 4,
    require_2d: bool = False,
) -> NDArray:
    """Validate an image numpy array.

    Args:
        image: Image array to validate.
        min_channels: Minimum required channels.
        max_channels: Maximum allowed channels.
        require_2d: Require grayscale (2D) image.

    Returns:
        Validated image array.

    Raises:
        InvalidImageError: If image is invalid.

    Example:
        >>> validated = validate_image_array(image)
    """
    import numpy as np  # pylint: disable=import-outside-toplevel

    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.exceptions import InvalidImageError

    if not isinstance(image, np.ndarray):
        raise InvalidImageError("Image must be a numpy array")

    if image.size == 0:
        raise InvalidImageError("Image array is empty")

    if require_2d:
        if len(image.shape) != 2:
            raise InvalidImageError("Image must be 2D (grayscale)")
    else:
        if len(image.shape) not in (2, 3):
            raise InvalidImageError(
                f"Image must be 2D or 3D, got shape: {image.shape}"
            )

        if len(image.shape) == 3:
            channels = image.shape[2]
            if channels < min_channels or channels > max_channels:
                raise InvalidImageError(
                    f"Image must have {min_channels}-{max_channels} channels, "
                    f"got {channels}"
                )

    return image
