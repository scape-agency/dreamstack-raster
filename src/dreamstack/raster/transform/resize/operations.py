# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Resize Operations
=================

Comprehensive image resizing and scaling utilities.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

import numpy as np

# =============================================================================
# Type Checking Imports
# =============================================================================

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel
    from numpy.typing import NDArray


Interpolation = Literal["nearest", "linear", "cubic", "lanczos", "area"]


class ResizeMethod(str, Enum):
    """Resize fitting method.

    Attributes:
        STRETCH: Stretch to exact dimensions (distorts aspect ratio).
        FIT: Scale to fit within dimensions (maintains aspect).
        FILL: Scale to fill dimensions, cropping if needed.
        PAD: Scale to fit and pad to exact dimensions.
    """

    STRETCH = "stretch"
    FIT = "fit"
    FILL = "fill"
    PAD = "pad"


@dataclass
class ResizeConfig:
    """Configuration for resize operations.

    Attributes:
        interpolation: Interpolation method.
        preserve_aspect: Maintain aspect ratio.
        divisible_by: Ensure dimensions are divisible by this value.
    """

    interpolation: Interpolation = "lanczos"
    preserve_aspect: bool = True
    divisible_by: int | None = None


def _get_cv2_interpolation(method: Interpolation) -> int:
    """Map interpolation string to OpenCV constant."""
    import cv2  # pylint: disable=import-outside-toplevel

    # pylint: disable=no-member
    mapping = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
        "area": cv2.INTER_AREA,
    }
    return mapping.get(method, cv2.INTER_LINEAR)
    # pylint: enable=no-member


def resize(
    image: NDArray[np.uint8],
    size: tuple[int, int],
    *,
    interpolation: Interpolation = "lanczos",
) -> NDArray[np.uint8]:
    """Resize image to exact dimensions.

    Args:
        image: Input image.
        size: Target (width, height).
        interpolation: Interpolation method.

    Returns:
        Resized image.

    Example:
        >>> resized = resize(image, (800, 600))
    """
    import cv2  # pylint: disable=import-outside-toplevel

    interp = _get_cv2_interpolation(interpolation)
    return np.asarray(
        cv2.resize(
            image, size, interpolation=interp
        ),  # pylint: disable=no-member
        dtype=np.uint8,
    )


def scale(
    image: NDArray[np.uint8],
    factor: float,
    *,
    interpolation: Interpolation = "lanczos",
) -> NDArray[np.uint8]:
    """Scale image by a factor.

    Args:
        image: Input image.
        factor: Scale factor (e.g., 0.5 for half size, 2.0 for double).
        interpolation: Interpolation method.

    Returns:
        Scaled image.

    Example:
        >>> half = scale(image, 0.5)
        >>> double = scale(image, 2.0)
    """
    h, w = image.shape[:2]
    new_size = (int(w * factor), int(h * factor))
    return resize(image, new_size, interpolation=interpolation)


def fit(
    image: NDArray[np.uint8],
    max_size: tuple[int, int],
    *,
    interpolation: Interpolation = "lanczos",
) -> NDArray[np.uint8]:
    """Fit image within maximum dimensions, preserving aspect ratio.

    The image is scaled down to fit entirely within the given dimensions.
    Will not upscale smaller images.

    Args:
        image: Input image.
        max_size: Maximum (width, height).
        interpolation: Interpolation method.

    Returns:
        Fitted image.

    Example:
        >>> fitted = fit(image, (1920, 1080))
    """
    h, w = image.shape[:2]
    max_w, max_h = max_size

    if w <= max_w and h <= max_h:
        return image.copy()

    scale_factor = min(max_w / w, max_h / h)
    new_size = (int(w * scale_factor), int(h * scale_factor))

    return resize(image, new_size, interpolation=interpolation)


def fill(
    image: NDArray[np.uint8],
    target_size: tuple[int, int],
    *,
    interpolation: Interpolation = "lanczos",
) -> NDArray[np.uint8]:
    """Fill target dimensions, cropping if necessary.

    The image is scaled to completely fill the target area,
    then center-cropped to exact dimensions.

    Args:
        image: Input image.
        target_size: Target (width, height).
        interpolation: Interpolation method.

    Returns:
        Filled and cropped image.

    Example:
        >>> filled = fill(image, (1920, 1080))
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size

    # Scale to fill (larger dimension matches, may overflow)
    scale_factor = max(target_w / w, target_h / h)
    scaled_w = int(w * scale_factor)
    scaled_h = int(h * scale_factor)

    scaled = resize(image, (scaled_w, scaled_h), interpolation=interpolation)

    # Center crop
    x_offset = (scaled_w - target_w) // 2
    y_offset = (scaled_h - target_h) // 2

    return scaled[
        y_offset : y_offset + target_h, x_offset : x_offset + target_w
    ]


def resize_to_width(
    image: NDArray[np.uint8],
    target_width: int,
    *,
    interpolation: Interpolation = "lanczos",
) -> NDArray[np.uint8]:
    """Resize image to target width, preserving aspect ratio.

    Args:
        image: Input image.
        target_width: Target width in pixels.
        interpolation: Interpolation method.

    Returns:
        Resized image.

    Example:
        >>> resized = resize_to_width(image, 1920)
    """
    h, w = image.shape[:2]
    scale_factor = target_width / w
    target_height = int(h * scale_factor)

    return resize(
        image, (target_width, target_height), interpolation=interpolation
    )


def resize_to_aspect(
    image: NDArray[np.uint8],
    target_width: int,
    aspect_ratio: tuple[int, int] = (16, 9),
    *,
    divisible_by: int = 1,
    interpolation: Interpolation = "lanczos",
) -> NDArray[np.uint8]:
    """Resize image to target width with specific aspect ratio.

    Useful for preparing images for AI models or video output.

    Args:
        image: Input image.
        target_width: Target width in pixels.
        aspect_ratio: Target aspect ratio as (width, height).
        divisible_by: Ensure dimensions are divisible by this value.
        interpolation: Interpolation method.

    Returns:
        Resized and cropped/padded image.

    Example:
        >>> # Resize to 1920x1080 (16:9 aspect)
        >>> result = resize_to_aspect(image, 1920, (16, 9))
        >>>
        >>> # Resize for AI model (dimensions divisible by 64)
        >>> result = resize_to_aspect(image, 1024, (16, 9), divisible_by=64)
    """
    # pylint: disable=import-outside-toplevel
    import cv2

    ar_w, ar_h = aspect_ratio
    target_height = int(target_width * ar_h / ar_w)

    # Adjust for divisibility
    if divisible_by > 1:
        target_width = (target_width // divisible_by) * divisible_by
        target_height = (target_height // divisible_by) * divisible_by

    interp = _get_cv2_interpolation(interpolation)
    return np.asarray(
        cv2.resize(
            image, (target_width, target_height), interpolation=interp
        ),  # pylint: disable=no-member
        dtype=np.uint8,
    )


def resize_for_ai(
    image: NDArray[np.uint8],
    target_width: int = 1024,
    *,
    divisible_by: int = 64,
    aspect_ratio: tuple[int, int] = (16, 9),
    interpolation: Interpolation = "lanczos",
) -> NDArray[np.uint8]:
    """Resize image for AI/ML model input.

    Ensures dimensions are divisible by a power of 2 for neural networks.

    Args:
        image: Input image.
        target_width: Target width.
        divisible_by: Ensure dimensions divisible by this (typically 8, 16, 32, or 64).
        aspect_ratio: Target aspect ratio.
        interpolation: Interpolation method.

    Returns:
        Resized image with AI-compatible dimensions.

    Example:
        >>> # Prepare for Stable Diffusion (768x512, divisible by 8)
        >>> ai_ready = resize_for_ai(image, 768, divisible_by=8, aspect_ratio=(3, 2))
    """
    return resize_to_aspect(
        image,
        target_width,
        aspect_ratio,
        divisible_by=divisible_by,
        interpolation=interpolation,
    )


def fit_to_dimensions(
    image: NDArray[np.uint8],
    max_width: int,
    max_height: int,
    *,
    interpolation: Interpolation = "lanczos",
    allow_upscale: bool = False,
) -> NDArray[np.uint8]:
    """Fit image within maximum dimensions, preserving aspect ratio.

    The image will be scaled down (or optionally up) to fit within
    the specified bounds while maintaining its aspect ratio.

    Args:
        image: Input image.
        max_width: Maximum allowed width.
        max_height: Maximum allowed height.
        interpolation: Interpolation method.
        allow_upscale: Allow scaling up smaller images.

    Returns:
        Fitted image.

    Example:
        >>> # Fit within 1920x1080 bounds
        >>> fitted = fit_to_dimensions(image, 1920, 1080)
    """
    h, w = image.shape[:2]

    # Calculate scale factors
    scale_w = max_width / w
    scale_h = max_height / h
    scale_factor = min(scale_w, scale_h)

    # Don't upscale unless allowed
    if not allow_upscale and scale_factor > 1.0:
        return image.copy()

    new_width = int(w * scale_factor)
    new_height = int(h * scale_factor)

    return resize(image, (new_width, new_height), interpolation=interpolation)


def thumbnail(
    image: NDArray[np.uint8],
    max_size: int = 256,
    *,
    interpolation: Interpolation = "area",
) -> NDArray[np.uint8]:
    """Create thumbnail with maximum dimension constraint.

    Args:
        image: Input image.
        max_size: Maximum width or height.
        interpolation: Interpolation method (area is best for downscaling).

    Returns:
        Thumbnail image.

    Example:
        >>> thumb = thumbnail(image, max_size=128)
    """
    return fit_to_dimensions(
        image,
        max_size,
        max_size,
        interpolation=interpolation,
        allow_upscale=False,
    )


def downscale(
    image: NDArray[np.uint8],
    max_size: int,
    *,
    preserve_aspect: bool = True,
    interpolation: Interpolation = "area",
) -> NDArray[np.uint8]:
    """Downscale image to fit within maximum size.

    Only scales down, never up. Uses INTER_AREA for best quality.

    Args:
        image: Input image.
        max_size: Maximum width or height.
        preserve_aspect: Keep original aspect ratio.
        interpolation: Interpolation method.

    Returns:
        Downscaled image (or original if already smaller).

    Example:
        >>> smaller = downscale(image, max_size=1024)
    """
    h, w = image.shape[:2]

    if max(w, h) <= max_size:
        return image.copy()

    if preserve_aspect:
        return fit_to_dimensions(
            image,
            max_size,
            max_size,
            interpolation=interpolation,
            allow_upscale=False,
        )
    else:
        return resize(image, (max_size, max_size), interpolation=interpolation)


def upscale(
    image: NDArray[np.uint8],
    scale_factor: float = 2.0,
    *,
    interpolation: Interpolation = "lanczos",
) -> NDArray[np.uint8]:
    """Upscale image by a factor.

    For AI-based upscaling, see the upscale module.

    Args:
        image: Input image.
        scale_factor: Scale factor (> 1.0 for upscaling).
        interpolation: Interpolation method.

    Returns:
        Upscaled image.

    Example:
        >>> upscaled = upscale(image, scale_factor=2.0)
    """
    h, w = image.shape[:2]
    new_width = int(w * scale_factor)
    new_height = int(h * scale_factor)

    return resize(image, (new_width, new_height), interpolation=interpolation)


def pad_to_aspect(
    image: NDArray[np.uint8],
    aspect_ratio: tuple[int, int] = (16, 9),
    *,
    pad_color: tuple[int, int, int] = (0, 0, 0),
    position: Literal["center", "top", "bottom", "left", "right"] = "center",
) -> NDArray[np.uint8]:
    """Pad image to achieve target aspect ratio.

    Adds black bars (letterbox/pillarbox) to reach the target aspect.

    Args:
        image: Input image.
        aspect_ratio: Target aspect ratio as (width, height).
        pad_color: Color for padding (BGR).
        position: Position of original image in padded result.

    Returns:
        Padded image with target aspect ratio.

    Example:
        >>> # Add letterbox to make 16:9
        >>> letterboxed = pad_to_aspect(image, (16, 9))
    """
    h, w = image.shape[:2]
    ar_w, ar_h = aspect_ratio

    current_ar = w / h
    target_ar = ar_w / ar_h

    if abs(current_ar - target_ar) < 0.001:
        return image.copy()

    if current_ar > target_ar:
        # Too wide, add vertical padding
        new_height = int(w / target_ar)
        pad_total = new_height - h

        if position == "top":
            pad_top, _ = 0, pad_total
        elif position == "bottom":
            pad_top, _ = pad_total, 0
        else:  # center
            pad_top = pad_total // 2
            # pad_bottom = pad_total - pad_top  # unused

        pad_left = 0
        new_width = w
    else:
        # Too tall, add horizontal padding
        new_width = int(h * target_ar)
        pad_total = new_width - w

        if position == "left":
            pad_left, _ = 0, pad_total
        elif position == "right":
            pad_left, _ = pad_total, 0
        else:  # center
            pad_left = pad_total // 2
            # pad_right = pad_total - pad_left (unused, handled by new_width)

        pad_top = 0
        new_height = h

    # Create padded image
    channels = image.shape[2] if image.ndim == 3 else 1
    if channels > 1:
        result = np.full(
            (new_height, new_width, channels),
            pad_color[:channels],
            dtype=np.uint8,
        )
    else:
        result = np.full((new_height, new_width), pad_color[0], dtype=np.uint8)

    # Place original image
    result[pad_top : pad_top + h, pad_left : pad_left + w] = image

    return result
