# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Color Module
================================

Professional color management, color space conversions,
and ICC profile handling.

This module provides two levels of color operations:

1. **Array-based operations** (for image processing):
   - `rgb_to_hsv`, `hsv_to_rgb`, etc. - Vectorized numpy operations
   - Optimized for processing entire images or regions

2. **Single-color operations** (raster-owned lightweight models):
    - `RGBColorModel`, `HSLColorModel`, `HSVColorModel`, `CMYKColorModel`
    - `Color` and `Palette` for swatches, gradients, and palette utilities
    - Use the `bridge` submodule to convert between arrays and models

Example:
    >>> from dreamstack.raster.color import Color, rgb_to_hsv
    >>> import numpy as np
    >>>
    >>> # Array-based: process image regions
    >>> pixels = np.array([[0.5, 0.3, 0.2], [0.8, 0.6, 0.4]])
    >>> hsv_pixels = rgb_to_hsv(pixels)
    >>>
    >>> # Model-based: single color manipulation
    >>> color = Color(128, 64, 32)
    >>> lighter = color.lighten(0.2)
    >>> complement = color.complement()

"""

# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from dreamstack.raster.color.bridge import (
    array_to_rgb,
    arrays_to_rgb_list,
    convert_color_model,
    get_color_model,
    rgb_list_to_arrays,
    rgb_to_array,
)
from dreamstack.raster.color.convert import (
    cmyk_to_rgb,
    convert_color,
    gray_to_rgb,
    hsl_to_rgb,
    hsv_to_rgb,
    lab_to_lch,
    lab_to_rgb,
    lch_to_lab,
    oklab_to_oklch,
    oklab_to_rgb,
    oklch_to_oklab,
    rgb_to_cmyk,
    rgb_to_gray,
    rgb_to_hsl,
    rgb_to_hsv,
    rgb_to_lab,
    rgb_to_oklab,
    rgb_to_xyz,
    rgb_to_ycbcr,
    xyz_to_rgb,
    ycbcr_to_rgb,
)
from dreamstack.raster.color.models import (
    CMYKColorModel,
    HSLColorModel,
    HSVColorModel,
    RGBColorModel,
)
from dreamstack.raster.color.palette import (
    BLACK,
    BLUE,
    CYAN,
    GREEN,
    MAGENTA,
    RED,
    TRANSPARENT,
    WHITE,
    YELLOW,
    Color,
    Palette,
    create_gradient,
    extract_palette,
)
from dreamstack.raster.color.pipeline import (
    ensure,
    to_encoded,
    to_linear,
    to_premultiplied,
    to_straight,
    to_working_space,
)
from dreamstack.raster.color.profiles import (
    ColorSpaceType,
    ICCProfile,
    ProfileClass,
    RenderingIntent,
    convert_profile,
    embed_profile,
    get_profile_info,
    get_system_profiles,
    load_profile,
)
from dreamstack.raster.color.spaces import (
    ACES,
    ACES_WP,
    D50,
    D65,
    DCI_P3,
    ACEScg,
    AdobeRGB,
    ColorSpace,
    DisplayP3,
    GammaType,
    ProPhotoRGB,
    Rec709,
    Rec2020,
    Rec2100HLG,
    Rec2100PQ,
    convert_color_space,
    get_color_space,
    list_color_spaces,
    sRGB,
)
from dreamstack.raster.core.channel import (
    channel_to_grayscale_rgb,
    extract_channel,
    extract_rgb_arrays,
    isolate_channel,
    merge_channels,
    split_channels,
    swap_channels,
)

__all__: list[str] = [
    # ==========================================================================
    # Array-based conversions (numpy, for image processing)
    # ==========================================================================
    "rgb_to_hsv",
    "hsv_to_rgb",
    "rgb_to_hsl",
    "hsl_to_rgb",
    "rgb_to_lab",
    "lab_to_rgb",
    "rgb_to_cmyk",
    "cmyk_to_rgb",
    "rgb_to_xyz",
    "xyz_to_rgb",
    "gray_to_rgb",
    "rgb_to_gray",
    "rgb_to_oklab",
    "oklab_to_rgb",
    "lab_to_lch",
    "lch_to_lab",
    "oklab_to_oklch",
    "oklch_to_oklab",
    "rgb_to_ycbcr",
    "ycbcr_to_rgb",
    "convert_color",
    # ==========================================================================
    # Bridge functions (array <-> model)
    # ==========================================================================
    "array_to_rgb",
    "rgb_to_array",
    "arrays_to_rgb_list",
    "rgb_list_to_arrays",
    "convert_color_model",
    "get_color_model",
    # ==========================================================================
    # Color models
    # ==========================================================================
    "RGBColorModel",
    "HSLColorModel",
    "HSVColorModel",
    "CMYKColorModel",
    # ==========================================================================
    # Color spaces (raster-specific)
    # ==========================================================================
    "GammaType",
    "ColorSpace",
    "get_color_space",
    "list_color_spaces",
    "convert_color_space",
    "sRGB",
    "AdobeRGB",
    "ProPhotoRGB",
    "DisplayP3",
    "Rec709",
    "Rec2020",
    "Rec2100PQ",
    "Rec2100HLG",
    "ACES",
    "ACEScg",
    "DCI_P3",
    "D65",
    "D50",
    "ACES_WP",
    # ==========================================================================
    # ICC Profiles (raster-specific)
    # ==========================================================================
    "RenderingIntent",
    "ProfileClass",
    "ColorSpaceType",
    "ICCProfile",
    "load_profile",
    "embed_profile",
    "convert_profile",
    "get_system_profiles",
    "get_profile_info",
    # ==========================================================================
    # Palette (raster-specific)
    # ==========================================================================
    "Palette",
    "Color",
    "extract_palette",
    "create_gradient",
    # ==========================================================================
    # Color presets
    # ==========================================================================
    "BLACK",
    "WHITE",
    "RED",
    "GREEN",
    "BLUE",
    "YELLOW",
    "CYAN",
    "MAGENTA",
    "TRANSPARENT",
    # ==========================================================================
    # Channel operations (for ML and image analysis)
    # ==========================================================================
    "split_channels",
    "extract_channel",
    "merge_channels",
    "isolate_channel",
    "extract_rgb_arrays",
    "swap_channels",
    "channel_to_grayscale_rgb",
    # ==========================================================================
    # Typed color pipeline
    # ==========================================================================
    "ensure",
    "to_linear",
    "to_encoded",
    "to_premultiplied",
    "to_straight",
    "to_working_space",
]
