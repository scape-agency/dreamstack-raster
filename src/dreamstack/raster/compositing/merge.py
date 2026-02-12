# -*- coding: utf-8 -*-

"""
Merge Operations
================

Image merging and arithmetic operations: add, subtract, multiply, etc.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


MergeMode = Literal["add", "subtract", "multiply", "divide", "screen", "difference"]


def add(
    image_a: NDArray[np.uint8],
    image_b: NDArray[np.uint8],
    *,
    factor: float = 1.0,
    clamp: bool = True,
) -> NDArray[np.uint8]:
    """Add two images together.
    
    Performs pixel-wise addition: result = A + (B * factor)
    
    Args:
        image_a: First image (base).
        image_b: Second image (to add).
        factor: Multiplier for second image (0-1).
        clamp: If True, clamp results to 0-255.
    
    Returns:
        Added image.
    
    Example:
        >>> combined = add(base, overlay)
        >>> # Add at 50% strength
        >>> combined = add(base, light, factor=0.5)
    """
    # Convert to float for computation
    a = image_a.astype(np.float32)
    b = image_b.astype(np.float32) * factor
    
    result = a + b
    
    if clamp:
        result = np.clip(result, 0, 255)
    
    return result.astype(np.uint8)


def subtract(
    image_a: NDArray[np.uint8],
    image_b: NDArray[np.uint8],
    *,
    factor: float = 1.0,
    clamp: bool = True,
) -> NDArray[np.uint8]:
    """Subtract one image from another.
    
    Performs pixel-wise subtraction: result = A - (B * factor)
    
    Args:
        image_a: Base image.
        image_b: Image to subtract.
        factor: Multiplier for subtracted image.
        clamp: If True, clamp results to 0-255.
    
    Returns:
        Subtracted image.
    
    Example:
        >>> diff = subtract(image_a, image_b)
    """
    a = image_a.astype(np.float32)
    b = image_b.astype(np.float32) * factor
    
    result = a - b
    
    if clamp:
        result = np.clip(result, 0, 255)
    
    return result.astype(np.uint8)


def multiply(
    image_a: NDArray[np.uint8],
    image_b: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    """Multiply two images.
    
    Performs pixel-wise multiplication (normalized).
    Result is always darker or equal to inputs.
    
    Args:
        image_a: First image.
        image_b: Second image.
    
    Returns:
        Multiplied image.
    
    Example:
        >>> darkened = multiply(image, shadow_mask)
    """
    a = image_a.astype(np.float32) / 255.0
    b = image_b.astype(np.float32) / 255.0
    
    result = a * b * 255.0
    
    return np.clip(result, 0, 255).astype(np.uint8)


def divide(
    image_a: NDArray[np.uint8],
    image_b: NDArray[np.uint8],
    *,
    epsilon: float = 1e-6,
) -> NDArray[np.uint8]:
    """Divide one image by another.
    
    Performs pixel-wise division. Used for color correction
    and removing lighting variations.
    
    Args:
        image_a: Numerator image.
        image_b: Denominator image.
        epsilon: Small value to prevent division by zero.
    
    Returns:
        Divided image.
    
    Example:
        >>> corrected = divide(image, light_pattern)
    """
    a = image_a.astype(np.float32)
    b = image_b.astype(np.float32) + epsilon
    
    result = (a / b) * 128.0  # Normalize to mid-gray
    
    return np.clip(result, 0, 255).astype(np.uint8)


def screen(
    image_a: NDArray[np.uint8],
    image_b: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    """Screen blend two images.
    
    Inverse of multiply. Result is always lighter or equal.
    Formula: 1 - (1-A) * (1-B)
    
    Args:
        image_a: First image.
        image_b: Second image.
    
    Returns:
        Screen blended image.
    
    Example:
        >>> lightened = screen(image, light_effect)
    """
    a = image_a.astype(np.float32) / 255.0
    b = image_b.astype(np.float32) / 255.0
    
    result = 1.0 - (1.0 - a) * (1.0 - b)
    
    return (result * 255.0).astype(np.uint8)


def difference(
    image_a: NDArray[np.uint8],
    image_b: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    """Calculate absolute difference between images.
    
    Returns the absolute difference at each pixel.
    Useful for comparison and detecting changes.
    
    Args:
        image_a: First image.
        image_b: Second image.
    
    Returns:
        Difference image.
    
    Example:
        >>> changes = difference(before, after)
    """
    a = image_a.astype(np.int16)
    b = image_b.astype(np.int16)
    
    result = np.abs(a - b)
    
    return result.astype(np.uint8)


def average(
    image_a: NDArray[np.uint8],
    image_b: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    """Average two images.
    
    Simple 50/50 blend of two images.
    
    Args:
        image_a: First image.
        image_b: Second image.
    
    Returns:
        Averaged image.
    """
    a = image_a.astype(np.float32)
    b = image_b.astype(np.float32)
    
    result = (a + b) / 2.0
    
    return result.astype(np.uint8)


def maximum(
    image_a: NDArray[np.uint8],
    image_b: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    """Take maximum of two images per-pixel.
    
    For each pixel, takes the brighter value from either image.
    
    Args:
        image_a: First image.
        image_b: Second image.
    
    Returns:
        Maximum image.
    """
    return np.maximum(image_a, image_b)


def minimum(
    image_a: NDArray[np.uint8],
    image_b: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    """Take minimum of two images per-pixel.
    
    For each pixel, takes the darker value from either image.
    
    Args:
        image_a: First image.
        image_b: Second image.
    
    Returns:
        Minimum image.
    """
    return np.minimum(image_a, image_b)


def merge(
    image_a: NDArray[np.uint8],
    image_b: NDArray[np.uint8],
    mode: MergeMode = "add",
    *,
    factor: float = 1.0,
) -> NDArray[np.uint8]:
    """Merge two images using specified mode.
    
    Convenience function for applying different merge operations.
    
    Args:
        image_a: First image.
        image_b: Second image.
        mode: Merge mode (add, subtract, multiply, divide, screen, difference).
        factor: Factor for add/subtract modes.
    
    Returns:
        Merged image.
    
    Example:
        >>> result = merge(a, b, mode="multiply")
    """
    operations = {
        "add": lambda: add(image_a, image_b, factor=factor),
        "subtract": lambda: subtract(image_a, image_b, factor=factor),
        "multiply": lambda: multiply(image_a, image_b),
        "divide": lambda: divide(image_a, image_b),
        "screen": lambda: screen(image_a, image_b),
        "difference": lambda: difference(image_a, image_b),
    }
    
    if mode not in operations:
        raise ValueError(f"Unknown merge mode: {mode}")
    
    return operations[mode]()


def over(
    foreground: NDArray[np.uint8],
    background: NDArray[np.uint8],
    *,
    premultiplied: bool = False,
) -> NDArray[np.uint8]:
    """Composite foreground over background using alpha.
    
    Standard "over" compositing operation using alpha channel.
    
    Args:
        foreground: Foreground image (must have alpha channel).
        background: Background image.
        premultiplied: If True, foreground is already premultiplied.
    
    Returns:
        Composited image.
    
    Example:
        >>> result = over(logo_with_alpha, photo)
    """
    if len(foreground.shape) != 3 or foreground.shape[2] != 4:
        raise ValueError("Foreground must have alpha channel (RGBA)")
    
    # Ensure background has alpha
    if len(background.shape) == 2:
        background = np.stack([background] * 3, axis=-1)
    if background.shape[2] == 3:
        bg_alpha = np.full((*background.shape[:2], 1), 255, dtype=np.uint8)
        background = np.concatenate([background, bg_alpha], axis=-1)
    
    fg = foreground.astype(np.float32) / 255.0
    bg = background.astype(np.float32) / 255.0
    
    fg_alpha = fg[:, :, 3:4]
    bg_alpha = bg[:, :, 3:4]
    
    if not premultiplied:
        fg_rgb = fg[:, :, :3] * fg_alpha
    else:
        fg_rgb = fg[:, :, :3]
    
    bg_rgb = bg[:, :, :3] * bg_alpha
    
    # Over operation
    out_alpha = fg_alpha + bg_alpha * (1 - fg_alpha)
    out_alpha = np.clip(out_alpha, 1e-6, 1.0)  # Prevent division by zero
    
    out_rgb = (fg_rgb + bg_rgb * (1 - fg_alpha)) / out_alpha
    
    result = np.concatenate([out_rgb, out_alpha], axis=-1)
    result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
    
    return result
