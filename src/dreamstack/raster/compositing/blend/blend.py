# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Blend Function
==============

Apply blend modes between two images.

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
    # pylint: disable=import-outside-toplevel
    from numpy.typing import NDArray

# pylint: disable=wrong-import-position
from dreamstack.raster.compositing.blend.blend_mode import BlendMode

# pylint: enable=wrong-import-position


def blend(
    base: NDArray[np.uint8],
    overlay: NDArray[np.uint8],
    mode: BlendMode = BlendMode.NORMAL,
    *,
    opacity: float = 1.0,
) -> NDArray[np.uint8]:
    """Blend two images using the specified blend mode.

    Args:
        base: Base image (bottom layer).
        overlay: Overlay image (top layer).
        mode: Blend mode to apply.
        opacity: Overlay opacity (0.0-1.0).

    Returns:
        Blended image.

    Example:
        >>> result = blend(background, foreground, BlendMode.MULTIPLY)
        >>> result = blend(photo, texture, BlendMode.OVERLAY, opacity=0.5)
    """
    # Ensure both images have same shape
    if base.shape[:2] != overlay.shape[:2]:
        overlay = cv2.resize(overlay, (base.shape[1], base.shape[0]))  # type: ignore[assignment]

    # Work with float for precision
    base_f = base.astype(np.float32) / 255.0
    overlay_f = overlay.astype(np.float32) / 255.0

    # Get RGB channels (handle grayscale and alpha)
    if base_f.ndim == 2:
        base_rgb = np.stack([base_f] * 3, axis=-1)
    elif base_f.shape[2] >= 3:
        base_rgb = base_f[:, :, :3]
    else:
        base_rgb = np.stack([base_f[:, :, 0]] * 3, axis=-1)

    if overlay_f.ndim == 2:
        overlay_rgb = np.stack([overlay_f] * 3, axis=-1)
    elif overlay_f.shape[2] >= 3:
        overlay_rgb = overlay_f[:, :, :3]
    else:
        overlay_rgb = np.stack([overlay_f[:, :, 0]] * 3, axis=-1)

    # Apply blend mode
    result = _apply_blend_mode(base_rgb, overlay_rgb, mode)

    # Apply opacity
    if opacity < 1.0:
        result = base_rgb * (1 - opacity) + result * opacity

    # Clip and convert back
    result = np.clip(result * 255, 0, 255).astype(np.uint8)

    # Preserve alpha channel if present
    if base.ndim > 2 and base.shape[2] == 4:
        result_rgba = np.zeros(base.shape, dtype=np.uint8)
        result_rgba[:, :, :3] = result
        result_rgba[:, :, 3] = base[:, :, 3]
        return result_rgba

    return result


def _apply_blend_mode(
    base: NDArray[np.float32],
    overlay: NDArray[np.float32],
    mode: BlendMode,
) -> NDArray[np.float32]:
    """Apply blend mode calculation."""

    # Avoid division by zero
    eps = 1e-6

    if mode == BlendMode.NORMAL:
        return overlay

    elif mode == BlendMode.MULTIPLY:
        return base * overlay

    elif mode == BlendMode.SCREEN:
        return 1 - (1 - base) * (1 - overlay)

    elif mode == BlendMode.OVERLAY:
        low = 2 * base * overlay
        high = 1 - 2 * (1 - base) * (1 - overlay)
        return np.where(base < 0.5, low, high)

    elif mode == BlendMode.DARKEN:
        return np.minimum(base, overlay)

    elif mode == BlendMode.LIGHTEN:
        return np.maximum(base, overlay)

    elif mode == BlendMode.COLOR_DODGE:
        return np.minimum(1, base / (1 - overlay + eps))

    elif mode == BlendMode.COLOR_BURN:
        return 1 - np.minimum(1, (1 - base) / (overlay + eps))

    elif mode == BlendMode.HARD_LIGHT:
        low = 2 * base * overlay
        high = 1 - 2 * (1 - base) * (1 - overlay)
        return np.where(overlay < 0.5, low, high)

    elif mode == BlendMode.SOFT_LIGHT:
        return (1 - 2 * overlay) * base * base + 2 * overlay * base

    elif mode == BlendMode.DIFFERENCE:
        return np.abs(base - overlay)

    elif mode == BlendMode.EXCLUSION:
        return base + overlay - 2 * base * overlay

    elif mode == BlendMode.LINEAR_BURN:
        return np.maximum(0, base + overlay - 1)

    elif mode == BlendMode.LINEAR_DODGE:
        return np.minimum(1, base + overlay)

    elif mode == BlendMode.VIVID_LIGHT:
        dodge = np.minimum(1, base / (2 * (1 - overlay) + eps))
        burn = 1 - np.minimum(1, (1 - base) / (2 * overlay + eps))
        return np.where(overlay < 0.5, burn, dodge)

    elif mode == BlendMode.LINEAR_LIGHT:
        return np.clip(base + 2 * overlay - 1, 0, 1)

    elif mode == BlendMode.PIN_LIGHT:
        low = np.minimum(base, 2 * overlay)
        high = np.maximum(base, 2 * overlay - 1)
        return np.where(overlay < 0.5, low, high)

    elif mode == BlendMode.HARD_MIX:
        return np.where(base + overlay >= 1, 1.0, 0.0)

    elif mode == BlendMode.SUBTRACT:
        return np.maximum(0, base - overlay)

    elif mode == BlendMode.DIVIDE:
        return np.minimum(1, base / (overlay + eps))

    elif mode == BlendMode.DISSOLVE:
        # Random dissolve based on overlay alpha/intensity
        random_mask = np.random.random(base.shape[:2])
        mask = (random_mask < np.mean(overlay, axis=2))[:, :, np.newaxis]
        return np.where(mask, overlay, base)

    elif mode == BlendMode.DARKER_COLOR:
        # Compare luminosity
        base_lum = (
            0.299 * base[:, :, 0]
            + 0.587 * base[:, :, 1]
            + 0.114 * base[:, :, 2]
        )
        overlay_lum = (
            0.299 * overlay[:, :, 0]
            + 0.587 * overlay[:, :, 1]
            + 0.114 * overlay[:, :, 2]
        )
        mask = (overlay_lum < base_lum)[:, :, np.newaxis]
        return np.where(mask, overlay, base)

    elif mode == BlendMode.LIGHTER_COLOR:
        base_lum = (
            0.299 * base[:, :, 0]
            + 0.587 * base[:, :, 1]
            + 0.114 * base[:, :, 2]
        )
        overlay_lum = (
            0.299 * overlay[:, :, 0]
            + 0.587 * overlay[:, :, 1]
            + 0.114 * overlay[:, :, 2]
        )
        mask = (overlay_lum > base_lum)[:, :, np.newaxis]
        return np.where(mask, overlay, base)

    elif mode in (
        BlendMode.HUE,
        BlendMode.SATURATION,
        BlendMode.COLOR,
        BlendMode.LUMINOSITY,
    ):
        return _blend_hsl_mode(base, overlay, mode)

    else:
        return overlay


def _blend_hsl_mode(
    base: NDArray[np.float32],
    overlay: NDArray[np.float32],
    mode: BlendMode,
) -> NDArray[np.float32]:
    """Apply HSL-based blend modes."""
    # Convert to HLS (OpenCV uses HLS, not HSL)
    base_u8 = (base * 255).astype(np.uint8)
    overlay_u8 = (overlay * 255).astype(np.uint8)

    base_hls = cv2.cvtColor(base_u8, cv2.COLOR_RGB2HLS).astype(np.float32)
    overlay_hls = cv2.cvtColor(overlay_u8, cv2.COLOR_RGB2HLS).astype(
        np.float32
    )

    result_hls = base_hls.copy()

    if mode == BlendMode.HUE:
        result_hls[:, :, 0] = overlay_hls[:, :, 0]

    elif mode == BlendMode.SATURATION:
        result_hls[:, :, 2] = overlay_hls[:, :, 2]

    elif mode == BlendMode.COLOR:
        result_hls[:, :, 0] = overlay_hls[:, :, 0]
        result_hls[:, :, 2] = overlay_hls[:, :, 2]

    elif mode == BlendMode.LUMINOSITY:
        result_hls[:, :, 1] = overlay_hls[:, :, 1]

    result_u8 = cv2.cvtColor(result_hls.astype(np.uint8), cv2.COLOR_HLS2RGB)
    return result_u8.astype(np.float32) / 255.0
