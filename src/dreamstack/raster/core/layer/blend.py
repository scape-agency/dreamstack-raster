# -*- coding: utf-8 -*-

"""
Dreamstack Raster - Blend Functions
===================================

Blend mode application function.

"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from dreamstack.raster.core.layer.blend_mode import BlendMode


def apply_blend_mode(
    base: NDArray, blend: NDArray, mode: BlendMode
) -> NDArray:
    """
    Apply blend mode to combine two layers.

    Args:
        base: Base layer (normalized float, 0-1)
        blend: Top layer to blend
        mode: Blend mode

    Returns:
        Blended result (normalized float, 0-1)
    """
    # Ensure float type
    base = base.astype(np.float32)
    blend = blend.astype(np.float32)

    if mode == BlendMode.NORMAL:
        return blend

    elif mode == BlendMode.MULTIPLY:
        return base * blend

    elif mode == BlendMode.SCREEN:
        return 1 - (1 - base) * (1 - blend)

    elif mode == BlendMode.OVERLAY:
        mask = base < 0.5
        result = np.where(
            mask, 2 * base * blend, 1 - 2 * (1 - base) * (1 - blend)
        )
        return result

    elif mode == BlendMode.DARKEN:
        return np.minimum(base, blend)

    elif mode == BlendMode.LIGHTEN:
        return np.maximum(base, blend)

    elif mode == BlendMode.COLOR_DODGE:
        # Avoid division by zero
        divisor = np.maximum(1 - blend, 0.0001)
        return np.minimum(base / divisor, 1)

    elif mode == BlendMode.COLOR_BURN:
        # Avoid division by zero
        divisor = np.maximum(blend, 0.0001)
        return 1 - np.minimum((1 - base) / divisor, 1)

    elif mode == BlendMode.HARD_LIGHT:
        mask = blend < 0.5
        result = np.where(
            mask, 2 * base * blend, 1 - 2 * (1 - base) * (1 - blend)
        )
        return result

    elif mode == BlendMode.SOFT_LIGHT:
        # Pegtop soft light
        return (1 - 2 * blend) * base * base + 2 * blend * base

    elif mode == BlendMode.DIFFERENCE:
        return np.abs(base - blend)

    elif mode == BlendMode.EXCLUSION:
        return base + blend - 2 * base * blend

    elif mode == BlendMode.SUBTRACT:
        return np.maximum(base - blend, 0)

    elif mode == BlendMode.DIVIDE:
        # Avoid division by zero
        divisor = np.maximum(blend, 0.0001)
        return np.minimum(base / divisor, 1)

    elif mode == BlendMode.LINEAR_BURN:
        return np.maximum(base + blend - 1, 0)

    elif mode == BlendMode.LINEAR_DODGE:
        return np.minimum(base + blend, 1)

    elif mode == BlendMode.VIVID_LIGHT:
        mask = blend < 0.5
        # Color burn for dark, color dodge for light
        burn_divisor = np.maximum(2 * blend, 0.0001)
        burn = 1 - np.minimum((1 - base) / burn_divisor, 1)
        dodge_divisor = np.maximum(1 - 2 * (blend - 0.5), 0.0001)
        dodge = np.minimum(base / dodge_divisor, 1)
        return np.where(mask, burn, dodge)

    elif mode == BlendMode.LINEAR_LIGHT:
        return np.clip(base + 2 * blend - 1, 0, 1)

    elif mode == BlendMode.PIN_LIGHT:
        mask1 = blend < 0.5
        result = np.where(
            mask1,
            np.minimum(base, 2 * blend),
            np.maximum(base, 2 * (blend - 0.5)),
        )
        return result

    elif mode == BlendMode.HARD_MIX:
        return np.where(base + blend >= 1, 1, 0).astype(np.float32)

    elif mode == BlendMode.DISSOLVE:
        # Random dissolve based on blend alpha
        random_mask = np.random.random(base.shape) < blend
        return np.where(random_mask, blend, base)

    # For color component modes, we need to work in HSL
    elif mode in (
        BlendMode.HUE,
        BlendMode.SATURATION,
        BlendMode.COLOR,
        BlendMode.LUMINOSITY,
    ):
        from dreamstack.raster.color.convert import hsl_to_rgb, rgb_to_hsl

        # Convert to HSL
        base_hsl = rgb_to_hsl(base[:, :, :3])
        blend_hsl = rgb_to_hsl(blend[:, :, :3])

        if mode == BlendMode.HUE:
            result_hsl = np.stack(
                [blend_hsl[:, :, 0], base_hsl[:, :, 1], base_hsl[:, :, 2]],
                axis=2,
            )
        elif mode == BlendMode.SATURATION:
            result_hsl = np.stack(
                [base_hsl[:, :, 0], blend_hsl[:, :, 1], base_hsl[:, :, 2]],
                axis=2,
            )
        elif mode == BlendMode.COLOR:
            result_hsl = np.stack(
                [blend_hsl[:, :, 0], blend_hsl[:, :, 1], base_hsl[:, :, 2]],
                axis=2,
            )
        else:  # LUMINOSITY
            result_hsl = np.stack(
                [base_hsl[:, :, 0], base_hsl[:, :, 1], blend_hsl[:, :, 2]],
                axis=2,
            )

        result = hsl_to_rgb(result_hsl)

        # Preserve alpha if present
        if base.shape[2] == 4:
            result = np.concatenate([result, base[:, :, 3:4]], axis=2)

        return result

    else:
        # Default to normal blend
        return blend
