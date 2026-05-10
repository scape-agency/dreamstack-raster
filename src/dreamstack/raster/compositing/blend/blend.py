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

# =============================================================================
# Type Checking Imports
# =============================================================================

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel
    from numpy.typing import NDArray

# pylint: disable=wrong-import-position
from dreamstack.raster.core.layer.blend_mode import BlendMode

# pylint: enable=wrong-import-position


def blend(
    base: NDArray,
    overlay: NDArray,
    mode: BlendMode = BlendMode.NORMAL,
    *,
    opacity: float = 1.0,
) -> NDArray:
    """Blend two images using the specified blend mode.

    Operates internally on float32 RGB in ``[0, 1]`` (Photoshop-style
    color blending) and preserves the input dtype on output. Accepts
    ``uint8``, ``uint16``, or float arrays. The base image's alpha
    channel (if any) is carried through unchanged.

    Args:
        base: Base image (bottom layer). 2-D, RGB, or RGBA.
        overlay: Overlay image (top layer). Same shape as base.
        mode: Blend mode to apply.
        opacity: Overlay opacity in ``[0, 1]``.

    Returns:
        Blended image with the same shape and dtype as ``base``.

    Example:
        >>> result = blend(background, foreground, BlendMode.MULTIPLY)
    """
    # Ensure both images have same spatial shape.
    if base.shape[:2] != overlay.shape[:2]:
        overlay = cv2.resize(overlay, (base.shape[1], base.shape[0]))

    base_f, base_scale = _to_unit_float(base)
    overlay_f, _ = _to_unit_float(overlay)

    base_rgb = _ensure_rgb(base_f)
    overlay_rgb = _ensure_rgb(overlay_f)

    blended = np.asarray(
        _apply_blend_mode(base_rgb, overlay_rgb, mode), dtype=np.float32
    )

    if opacity < 1.0:
        blended = base_rgb * (1.0 - opacity) + blended * opacity

    blended = np.clip(blended, 0.0, 1.0)

    # Reattach alpha if the base had one; otherwise return plain RGB.
    if base.ndim == 3 and base.shape[2] == 4:
        result_f = np.empty(base.shape, dtype=np.float32)
        result_f[:, :, :3] = blended
        result_f[:, :, 3] = base_f[:, :, 3]
    else:
        result_f = blended

    result_f = np.asarray(result_f, dtype=np.float32)

    return _from_unit_float(result_f, base.dtype, base_scale)


def blend_float(
    base: NDArray[np.float32],
    overlay: NDArray[np.float32],
    mode: BlendMode = BlendMode.NORMAL,
    *,
    opacity: float = 1.0,
) -> NDArray[np.float32]:
    """Float32 RGB-only blend kernel (no dtype handling, no alpha).

    Both inputs must be float32 RGB in ``[0, 1]`` with identical shapes.
    Returned array is float32 RGB in ``[0, 1]``.
    """
    blended = np.asarray(
        _apply_blend_mode(base, overlay, mode), dtype=np.float32
    )
    if opacity < 1.0:
        blended = base * (1.0 - opacity) + blended * opacity
    return np.asarray(np.clip(blended, 0.0, 1.0), dtype=np.float32)


def _to_unit_float(arr: NDArray) -> tuple[NDArray[np.float32], float]:
    """Cast an integer or float image to float32 in ``[0, 1]``.

    Returns the converted array and the scale that was applied (so the
    caller can invert it on the way back to the original dtype).
    """
    if np.issubdtype(arr.dtype, np.integer):
        scale = float(np.iinfo(arr.dtype).max)
        return arr.astype(np.float32) / scale, scale
    return arr.astype(np.float32, copy=False), 1.0


def _from_unit_float(
    arr: NDArray[np.float32],
    target_dtype: np.dtype,
    scale: float,
) -> NDArray:
    """Inverse of :func:`_to_unit_float`."""
    if np.issubdtype(target_dtype, np.integer):
        return np.asarray(np.clip(arr * scale, 0.0, scale), dtype=target_dtype)
    return arr.astype(target_dtype, copy=False)


def _ensure_rgb(arr: NDArray[np.float32]) -> NDArray[np.float32]:
    """Project an arbitrary 2-D / 3-channel / 4-channel image to RGB."""
    if arr.ndim == 2:
        return np.stack([arr] * 3, axis=-1)
    if arr.shape[2] >= 3:
        return arr[:, :, :3]
    return np.stack([arr[:, :, 0]] * 3, axis=-1)


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
        return np.asarray(
            (1 - 2 * overlay) * base * base + 2 * overlay * base,
            dtype=np.float32,
        )

    elif mode == BlendMode.DIFFERENCE:
        return np.abs(base - overlay)

    elif mode == BlendMode.EXCLUSION:
        return np.asarray(
            base + overlay - 2 * base * overlay, dtype=np.float32
        )

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
    """Apply HSL-based blend modes (float32 RGB in/out)."""
    # OpenCV's float HLS path expects RGB in [0, 1]; H is in [0, 360].
    base_hls = cv2.cvtColor(
        np.ascontiguousarray(base, dtype=np.float32), cv2.COLOR_RGB2HLS
    )
    overlay_hls = cv2.cvtColor(
        np.ascontiguousarray(overlay, dtype=np.float32), cv2.COLOR_RGB2HLS
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

    return np.asarray(
        cv2.cvtColor(result_hls, cv2.COLOR_HLS2RGB), dtype=np.float32
    )
