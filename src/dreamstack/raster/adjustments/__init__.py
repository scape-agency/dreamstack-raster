# -*- coding: utf-8 -*-

"""
Dreamstack Raster - Adjustments Module
======================================

Professional image adjustments including levels, curves,
brightness/contrast, color balance, and more.

"""

from __future__ import annotations

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
    gamma_rgb,
)
from dreamstack.raster.adjustments.clamp import gamma as gamma_per_channel
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
from dreamstack.raster.adjustments.tone import (
    dehaze,
    hdr_toning,
    shadows_highlights,
    split_toning,
    tone_curve,
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
]
