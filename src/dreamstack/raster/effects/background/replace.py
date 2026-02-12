# -*- coding: utf-8 -*-

"""
Background Replacement Operations
=================================

Replace backgrounds with colors, images, gradients, or effects.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


GradientDirection = Literal["horizontal", "vertical", "diagonal", "radial"]


@dataclass
class GradientConfig:
    """Configuration for gradient backgrounds.
    
    Attributes:
        start_color: Starting RGB color.
        end_color: Ending RGB color.
        direction: Gradient direction.
        center: Center point for radial gradients (normalized 0-1).
    """
    
    start_color: tuple[int, int, int] = (255, 255, 255)
    end_color: tuple[int, int, int] = (200, 200, 200)
    direction: GradientDirection = "vertical"
    center: tuple[float, float] = (0.5, 0.5)


def replace_background(
    rgba_image: NDArray[np.uint8],
    background: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    """Replace background of RGBA image with another image.
    
    Composites the RGBA foreground onto the provided background image.
    Background will be resized to match the foreground dimensions.
    
    Args:
        rgba_image: RGBA image with alpha channel (4 channels).
        background: Background image (RGB, 3 channels).
    
    Returns:
        RGB image with replaced background.
    
    Example:
        >>> from dreamstack.raster.effects.background import (
        ...     remove_background,
        ...     replace_background,
        ... )
        >>> rgba = remove_background(image)
        >>> result = replace_background(rgba, new_bg)
    """
    import cv2
    
    if rgba_image.ndim != 3 or rgba_image.shape[2] != 4:
        raise ValueError("Expected RGBA image with 4 channels")
    
    h, w = rgba_image.shape[:2]
    
    # Resize background if needed
    if background.shape[:2] != (h, w):
        background = cv2.resize(background, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Ensure background is 3 channels
    if background.ndim == 2:
        background = cv2.cvtColor(background, cv2.COLOR_GRAY2RGB)
    elif background.shape[2] == 4:
        background = background[:, :, :3]
    
    # Extract alpha and normalize
    alpha = rgba_image[:, :, 3:4].astype(np.float32) / 255.0
    fg = rgba_image[:, :, :3].astype(np.float32)
    bg = background.astype(np.float32)
    
    # Alpha blend
    result = fg * alpha + bg * (1 - alpha)
    
    return result.astype(np.uint8)


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
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), 
                          interpolation=cv2.INTER_LINEAR)
    
    # Create blurred background
    blurred = cv2.GaussianBlur(image, (blur_radius, blur_radius), 0)
    
    # Normalize mask for blending
    alpha = mask.astype(np.float32) / 255.0
    alpha = np.expand_dims(alpha, axis=-1)
    
    # Blend: foreground is sharp, background is blurred
    result = image.astype(np.float32) * alpha + blurred.astype(np.float32) * (1 - alpha)
    
    return result.astype(np.uint8)


def replace_background_with_gradient(
    rgba_image: NDArray[np.uint8],
    config: GradientConfig | None = None,
    *,
    start_color: tuple[int, int, int] | None = None,
    end_color: tuple[int, int, int] | None = None,
    direction: GradientDirection | None = None,
) -> NDArray[np.uint8]:
    """Replace background with a color gradient.
    
    Creates a gradient background behind the RGBA foreground.
    
    Args:
        rgba_image: RGBA image with alpha channel.
        config: Optional GradientConfig.
        start_color: Starting gradient color.
        end_color: Ending gradient color.
        direction: Gradient direction.
    
    Returns:
        RGB image with gradient background.
    
    Example:
        >>> result = replace_background_with_gradient(
        ...     rgba,
        ...     start_color=(255, 200, 200),
        ...     end_color=(200, 200, 255),
        ...     direction="horizontal",
        ... )
    """
    cfg = config or GradientConfig()
    
    # Override with keyword arguments
    s_color = start_color or cfg.start_color
    e_color = end_color or cfg.end_color
    grad_dir = direction or cfg.direction
    
    if rgba_image.ndim != 3 or rgba_image.shape[2] != 4:
        raise ValueError("Expected RGBA image with 4 channels")
    
    h, w = rgba_image.shape[:2]
    
    # Create gradient
    gradient = _create_gradient(w, h, s_color, e_color, grad_dir, cfg.center)
    
    # Composite
    return replace_background(rgba_image, gradient)


def _create_gradient(
    width: int,
    height: int,
    start_color: tuple[int, int, int],
    end_color: tuple[int, int, int],
    direction: GradientDirection,
    center: tuple[float, float] = (0.5, 0.5),
) -> NDArray[np.uint8]:
    """Create a gradient image.
    
    Internal function to generate gradient backgrounds.
    """
    # Create coordinate grids
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xx, yy = np.meshgrid(x, y)
    
    # Calculate interpolation factor based on direction
    if direction == "horizontal":
        t = xx
    elif direction == "vertical":
        t = yy
    elif direction == "diagonal":
        t = (xx + yy) / 2
    elif direction == "radial":
        cx, cy = center
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        t = np.clip(dist / np.sqrt(2), 0, 1)
    else:
        t = yy  # Default to vertical
    
    # Expand dimensions for broadcasting
    t = np.expand_dims(t, axis=-1)
    
    # Interpolate colors
    start = np.array(start_color, dtype=np.float32)
    end = np.array(end_color, dtype=np.float32)
    
    gradient = start + t * (end - start)
    
    return gradient.astype(np.uint8)
