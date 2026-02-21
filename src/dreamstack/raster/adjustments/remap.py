# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Value Remapping Utilities
=========================

Remap grayscale and color values between different ranges.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class RemapConfig:
    """Configuration for value remapping.

    Attributes:
        input_min: Original minimum value.
        input_max: Original maximum value.
        output_min: Target minimum value.
        output_max: Target maximum value.
        clip: Clip values outside output range.
    """

    input_min: int = 0
    input_max: int = 255
    output_min: int = 0
    output_max: int = 255
    clip: bool = True


def remap_values(
    image: NDArray[np.uint8],
    input_range: tuple[int, int] = (0, 255),
    output_range: tuple[int, int] = (0, 255),
    *,
    clip: bool = True,
) -> NDArray[np.uint8]:
    """Remap image values from one range to another.

    Linearly maps values from the input range to the output range.
    Values outside the input range are extrapolated unless clipped.

    Args:
        image: Input image (any channel count).
        input_range: (min, max) of input values to map.
        output_range: (min, max) of output values.
        clip: Clip output to the output range.

    Returns:
        Remapped image.

    Example:
        >>> # Remap mid-range grays to full range
        >>> result = remap_values(image, (64, 192), (0, 255))
        >>>
        >>> # Compress dynamic range
        >>> result = remap_values(image, (0, 255), (32, 224))
    """
    in_min, in_max = input_range
    out_min, out_max = output_range

    # Handle edge case
    if in_max == in_min:
        return np.full_like(image, out_min)

    # Convert to float for calculation
    img_float = image.astype(np.float32)

    # Linear remapping
    scale = (out_max - out_min) / (in_max - in_min)
    result = (img_float - in_min) * scale + out_min

    if clip:
        result = np.clip(result, out_min, out_max)

    return result.astype(np.uint8)


def remap_grayscale(
    image: NDArray[np.uint8],
    input_range: tuple[int, int],
    output_range: tuple[int, int],
) -> NDArray[np.uint8]:
    """Remap grayscale values between ranges.

    Specialized function for grayscale images.
    Converts to grayscale if needed.

    Args:
        image: Input image (grayscale or color).
        input_range: (min, max) input values.
        output_range: (min, max) output values.

    Returns:
        Remapped grayscale image.

    Example:
        >>> # Make dark grays darker, light grays lighter
        >>> gray = remap_grayscale(image, (50, 200), (0, 255))
    """
    import cv2  # pylint: disable=import-outside-toplevel

    # Convert to grayscale if needed
    if image.ndim == 3:
        gray = np.asarray(
            cv2.cvtColor(
                image, cv2.COLOR_BGR2GRAY
            ),  # pylint: disable=no-member
            dtype=np.uint8,
        )
    else:
        gray = image.copy()

    return remap_values(gray, input_range, output_range)


def auto_remap(
    image: NDArray[np.uint8],
    output_range: tuple[int, int] = (0, 255),
    *,
    percentile: tuple[float, float] = (1.0, 99.0),
) -> NDArray[np.uint8]:
    """Automatically remap values using percentile-based input range.

    Automatically determines the input range based on percentiles
    to handle outliers and extend dynamic range.

    Args:
        image: Input image.
        output_range: Target output range.
        percentile: (low, high) percentiles for input range detection.

    Returns:
        Auto-remapped image.

    Example:
        >>> # Auto-stretch contrast
        >>> result = auto_remap(image)
        >>>
        >>> # Ignore outliers
        >>> result = auto_remap(image, percentile=(2, 98))
    """
    import cv2  # pylint: disable=import-outside-toplevel

    # Calculate percentiles
    if image.ndim == 3:
        gray = cv2.cvtColor(
            image, cv2.COLOR_BGR2GRAY
        )  # pylint: disable=no-member
    else:
        gray = image

    low_val = np.percentile(gray, percentile[0])
    high_val = np.percentile(gray, percentile[1])

    input_range = (int(low_val), int(high_val))

    return remap_values(image, input_range, output_range)


def invert_values(
    image: NDArray[np.uint8],
    max_value: int = 255,
) -> NDArray[np.uint8]:
    """Invert image values (negative effect).

    Args:
        image: Input image.
        max_value: Maximum value for inversion calculation.

    Returns:
        Inverted image.

    Example:
        >>> negative = invert_values(image)
    """
    return (max_value - image).astype(np.uint8)


def threshold_values(
    image: NDArray[np.uint8],
    threshold: int = 128,
    *,
    below_value: int = 0,
    above_value: int = 255,
) -> NDArray[np.uint8]:
    """Threshold image values to binary.

    Args:
        image: Input image.
        threshold: Threshold value.
        below_value: Value for pixels below threshold.
        above_value: Value for pixels at or above threshold.

    Returns:
        Thresholded image.

    Example:
        >>> binary = threshold_values(image, threshold=100)
    """
    return np.where(image >= threshold, above_value, below_value).astype(
        np.uint8
    )


def normalize_to_range(
    image: NDArray[np.uint8],
    output_range: tuple[int, int] = (0, 255),
) -> NDArray[np.uint8]:
    """Normalize image to use full output range.

    Maps the actual min/max values in the image to the output range.

    Args:
        image: Input image.
        output_range: Target output range.

    Returns:
        Normalized image.

    Example:
        >>> # Stretch to full range
        >>> normalized = normalize_to_range(image)
    """
    actual_min = int(image.min())
    actual_max = int(image.max())

    if actual_min == actual_max:
        return np.full_like(image, output_range[0])

    return remap_values(image, (actual_min, actual_max), output_range)


def gamma_correction(
    image: NDArray[np.uint8],
    gamma: float = 1.0,
) -> NDArray[np.uint8]:
    """Apply gamma correction to image.

    Gamma < 1 brightens, gamma > 1 darkens.

    Args:
        image: Input image.
        gamma: Gamma value.

    Returns:
        Gamma-corrected image.

    Example:
        >>> brighter = gamma_correction(image, gamma=0.7)
        >>> darker = gamma_correction(image, gamma=1.5)
    """
    if gamma <= 0:
        gamma = 0.01

    # Build lookup table for efficiency
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
    ).astype(np.uint8)

    import cv2  # pylint: disable=import-outside-toplevel

    return np.asarray(
        cv2.LUT(image, table), dtype=np.uint8
    )  # pylint: disable=no-member
