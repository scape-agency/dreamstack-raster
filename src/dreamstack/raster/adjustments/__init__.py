"""
Dreamstack Raster - Adjustments Module
======================================

Professional image adjustments including levels, curves,
brightness/contrast, color balance, and more.

"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from dreamstack.raster.adjustments.basic import (
    brightness,
    brightness_contrast,
    contrast,
    exposure,
    gamma,
    saturation,
    vibrance,
)
from dreamstack.raster.adjustments.black_white import (
    black_white,
    channel_mixer,
    desaturate,
    duotone,
    gradient_map,
    invert,
    sepia,
    threshold,
    tritone,
)
from dreamstack.raster.adjustments.clamp import (
    auto_gamma,
    clamp,
    clamp_normalized,
)
from dreamstack.raster.adjustments.clamp import gamma as gamma_per_channel
from dreamstack.raster.adjustments.clamp import gamma_rgb
from dreamstack.raster.adjustments.color_balance import (
    color_balance,
    hue_saturation,
    match_color,
    replace_color,
    selective_color,
)
from dreamstack.raster.adjustments.curves import (
    Curve,
    CurvePoint,
    apply_curve,
    create_curve,
    curves,
    linear_contrast,
    preset_curves,
    s_curve,
)
from dreamstack.raster.adjustments.levels import (
    auto_color,
    auto_contrast,
    auto_levels,
    equalize_histogram,
    input_levels,
    levels,
    output_levels,
)
from dreamstack.raster.adjustments.remap import (
    RemapConfig,
    auto_remap,
    gamma_correction,
    invert_values,
    normalize_to_range,
    remap_grayscale,
    remap_values,
    threshold_values,
)
from dreamstack.raster.adjustments.tone import (
    dehaze,
    hdr_toning,
    shadows_highlights,
    split_toning,
    tone_curve,
)


def apply_adjustment(
    data: NDArray[np.floating],
    adjustment_type: str,
    parameters: dict[str, Any],
) -> NDArray[np.floating]:
    """Apply an adjustment by name.

    Args:
        data: Input pixel data (normalized float 0-1).
        adjustment_type: Name of the adjustment to apply.
        parameters: Parameters for the adjustment.

    Returns:
        Adjusted pixel data.
    """
    adjustments_map = {
        "brightness": brightness,
        "contrast": contrast,
        "brightness_contrast": brightness_contrast,
        "exposure": exposure,
        "gamma": gamma,
        "vibrance": vibrance,
        "saturation": saturation,
        "levels": levels,
        "auto_levels": auto_levels,
        "curves": curves,
        "color_balance": color_balance,
        "hue_saturation": hue_saturation,
        "shadows_highlights": shadows_highlights,
        "invert": invert,
        "threshold": threshold,
        "black_white": black_white,
        "sepia": sepia,
    }

    func = adjustments_map.get(adjustment_type)
    if func is None:
        return data

    # Convert to uint8 for adjustment functions if needed
    if data.dtype in (np.float32, np.float64):
        data_uint8 = (np.clip(data, 0, 1) * 255).astype(np.uint8)
        result = func(data_uint8, **parameters)  # type: ignore[operator]
        return result.astype(np.float32) / 255.0

    return func(data, **parameters)  # type: ignore[operator]


__all__: list[str] = [
    # Basic
    "brightness",
    "contrast",
    "brightness_contrast",
    "exposure",
    "gamma",
    "vibrance",
    "saturation",
    # Levels
    "levels",
    "auto_levels",
    "auto_contrast",
    "auto_color",
    "input_levels",
    "output_levels",
    "equalize_histogram",
    # Curves
    "curves",
    "apply_curve",
    "create_curve",
    "CurvePoint",
    "Curve",
    "s_curve",
    "linear_contrast",
    "preset_curves",
    # Color Balance
    "color_balance",
    "hue_saturation",
    "selective_color",
    "replace_color",
    "match_color",
    # Tone
    "shadows_highlights",
    "hdr_toning",
    "tone_curve",
    "split_toning",
    "dehaze",
    # Black & White
    "black_white",
    "desaturate",
    "channel_mixer",
    "gradient_map",
    "duotone",
    "tritone",
    "sepia",
    "invert",
    "threshold",
    # Remap
    "remap_values",
    "remap_grayscale",
    "auto_remap",
    "invert_values",
    "threshold_values",
    "normalize_to_range",
    "gamma_correction",
    "RemapConfig",
    # Clamp & Gamma
    "clamp",
    "clamp_normalized",
    "gamma_per_channel",
    "gamma_rgb",
    "auto_gamma",
    # Apply adjustment
    "apply_adjustment",
]
