"""
Clamp and Gamma Adjustments
===========================

Pixel value clamping and gamma correction operations.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def clamp(
    image: NDArray[np.uint8],
    min_value: float = 0.0,
    max_value: float = 255.0,
    *,
    clamp_alpha: bool = False,
) -> NDArray[np.uint8]:
    """Clamp pixel values between min and max limits.

    Restricts all pixel values to be within the specified range.
    Values below min_value are set to min_value, values above
    max_value are set to max_value.

    Args:
        image: Input image (uint8 or float32).
        min_value: Minimum pixel value (0-255 for uint8).
        max_value: Maximum pixel value (0-255 for uint8).
        clamp_alpha: If True, also clamp the alpha channel.

    Returns:
        Clamped image.

    Example:
        >>> # Clamp to avoid extreme values
        >>> clamped = clamp(image, 10, 245)
        >>>
        >>> # Clamp to specific range
        >>> clamped = clamp(image, 50, 200)
    """
    result = image.copy()

    # Handle RGBA images
    if len(result.shape) == 3 and result.shape[2] == 4 and not clamp_alpha:
        # Clamp RGB only, preserve alpha
        result[:, :, :3] = np.clip(result[:, :, :3], min_value, max_value)
    else:
        # Clamp all channels
        result = np.clip(result, min_value, max_value)

    return result.astype(image.dtype)


def clamp_normalized(
    image: NDArray[np.float32],
    min_value: float = 0.0,
    max_value: float = 1.0,
    *,
    clamp_alpha: bool = False,
) -> NDArray[np.float32]:
    """Clamp normalized (0-1) pixel values.

    Args:
        image: Input image (float32, 0-1 range).
        min_value: Minimum value (0-1).
        max_value: Maximum value (0-1).
        clamp_alpha: If True, also clamp the alpha channel.

    Returns:
        Clamped image.

    Example:
        >>> # Clamp HDR values to displayable range
        >>> clamped = clamp_normalized(hdr_image, 0.0, 1.0)
    """
    return clamp(image, min_value, max_value, clamp_alpha=clamp_alpha)


def gamma(
    image: NDArray[np.uint8],
    gamma_value: float,
    *,
    gamma_r: float | None = None,
    gamma_g: float | None = None,
    gamma_b: float | None = None,
    gamma_a: float = 1.0,
) -> NDArray[np.uint8]:
    """Apply gamma correction to an image.

    Gamma correction adjusts the luminance of an image.
    Values < 1.0 brighten the image, values > 1.0 darken it.

    Can apply uniform gamma to all channels or different values
    per channel.

    Args:
        image: Input image.
        gamma_value: Gamma value for all RGB channels.
        gamma_r: Override gamma for red channel.
        gamma_g: Override gamma for green channel.
        gamma_b: Override gamma for blue channel.
        gamma_a: Gamma for alpha channel (default 1.0, no change).

    Returns:
        Gamma-corrected image.

    Example:
        >>> # Brighten image
        >>> bright = gamma(image, 0.8)
        >>>
        >>> # Darken image
        >>> dark = gamma(image, 1.5)
        >>>
        >>> # Per-channel gamma
        >>> adjusted = gamma(image, 1.0, gamma_r=0.9, gamma_b=1.1)
    """
    # Set per-channel values
    g_r = gamma_r if gamma_r is not None else gamma_value
    g_g = gamma_g if gamma_g is not None else gamma_value
    g_b = gamma_b if gamma_b is not None else gamma_value

    # Convert to float for precision
    img_float = image.astype(np.float32) / 255.0

    # Apply gamma (power function)
    # Note: gamma correction is typically img^(1/gamma)
    if len(img_float.shape) == 3:
        channels = img_float.shape[2]

        if channels >= 3:
            img_float[:, :, 0] = np.power(img_float[:, :, 0], 1.0 / g_r)
            img_float[:, :, 1] = np.power(img_float[:, :, 1], 1.0 / g_g)
            img_float[:, :, 2] = np.power(img_float[:, :, 2], 1.0 / g_b)

        if channels == 4:
            img_float[:, :, 3] = np.power(img_float[:, :, 3], 1.0 / gamma_a)
    else:
        # Grayscale
        img_float = np.power(img_float, 1.0 / gamma_value)

    # Convert back to uint8
    result = np.clip(img_float * 255.0, 0, 255).astype(np.uint8)

    return result


def gamma_rgb(
    image: NDArray[np.uint8],
    gamma_r: float,
    gamma_g: float,
    gamma_b: float,
    gamma_a: float = 1.0,
) -> NDArray[np.uint8]:
    """Apply per-channel gamma correction.

    Convenient shorthand for applying different gamma values
    to each color channel.

    Args:
        image: Input image.
        gamma_r: Gamma for red channel.
        gamma_g: Gamma for green channel.
        gamma_b: Gamma for blue channel.
        gamma_a: Gamma for alpha channel (default 1.0).

    Returns:
        Gamma-corrected image.

    Example:
        >>> # Warm up image (boost red, reduce blue)
        >>> warm = gamma_rgb(image, 0.9, 1.0, 1.1)
    """
    return gamma(
        image,
        1.0,
        gamma_r=gamma_r,
        gamma_g=gamma_g,
        gamma_b=gamma_b,
        gamma_a=gamma_a,
    )


def auto_gamma(
    image: NDArray[np.uint8],
    target_mean: float = 0.5,
) -> NDArray[np.uint8]:
    """Automatically adjust gamma based on image brightness.

    Calculates gamma value to achieve target mean brightness.

    Args:
        image: Input image.
        target_mean: Target mean brightness (0-1, default 0.5).

    Returns:
        Gamma-corrected image.

    Example:
        >>> # Auto-adjust to mid-gray average
        >>> adjusted = auto_gamma(image)
        >>>
        >>> # Auto-adjust to brighter average
        >>> bright = auto_gamma(image, target_mean=0.6)
    """
    # Calculate current mean (normalized)
    if len(image.shape) == 3:
        # Use luminance for color images
        gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        current_mean = np.mean(gray) / 255.0
    else:
        current_mean = np.mean(image) / 255.0

    # Avoid division by zero
    if current_mean < 0.001:
        return image.copy()

    # Calculate gamma: target = current^(1/gamma)
    # So gamma = log(current) / log(target)
    if target_mean <= 0 or target_mean >= 1:
        return image.copy()

    gamma_value = np.log(current_mean) / np.log(target_mean)

    # Clamp to reasonable range
    gamma_value = np.clip(gamma_value, 0.1, 4.0)

    return gamma(image, gamma_value)
