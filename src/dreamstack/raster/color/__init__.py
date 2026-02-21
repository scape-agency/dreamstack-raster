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

2. **Model-based operations** (from dreamstack.color):
   - `RGB`, `HSL`, `HSV`, `CMYK` - Color model classes
   - `lighten`, `darken`, `saturate`, etc. - Single-color manipulation
   - `complementary`, `triadic`, etc. - Color harmony functions
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

# Re-export dreamstack.color models and functions for convenience
from dreamstack.color import (  # Models; Manipulation functions; Harmony functions; Distance & Comparison; Gradients; Generators; Validators; Utils; Additional conversions
    NAMED_COLORS,
    CMYKColorModel,
    HSLColorModel,
    HSVColorModel,
    RGBColorModel,
    adjust_hue,
    analogous,
    bezier_gradient,
    brighten,
    complement,
    complementary,
    contrast_ratio,
    darken,
    desaturate,
    dim,
    euclidean_distance,
    fade,
    get_named_color,
    golden_ratio_color,
    grayscale,
    hex_to_rgb,
    interpolate,
    invert,
    is_dark,
    is_light,
    is_similar,
    is_valid_css_color,
    is_valid_hex,
    is_valid_rgb,
    lighten,
    linear_gradient,
    luminance,
    manhattan_distance,
    mix,
    monochromatic,
    parse_css_color,
    passes_wcag_aa,
    passes_wcag_aaa,
    random_color,
    random_palette,
    rgb_to_hex,
    saturate,
    shades,
    split_complementary,
    square,
    tetradic,
    tints,
    tones,
    triadic,
)
from dreamstack.raster.color.bridge import (
    array_to_rgb,
    arrays_to_rgb_list,
    convert_color_model,
    get_color_model,
    rgb_list_to_arrays,
    rgb_to_array,
)
from dreamstack.raster.color.channels import (
    channel_to_grayscale_rgb,
    extract_channel,
    extract_rgb_arrays,
    isolate_channel,
    merge_channels,
    split_channels,
    swap_channels,
)

# Import from submodules (now directories with __init__.py)
from dreamstack.raster.color.convert import (
    cmyk_to_rgb,
    convert_color,
    gray_to_rgb,
    hsl_to_rgb,
    hsv_to_rgb,
    lab_to_rgb,
    rgb_to_cmyk,
    rgb_to_gray,
    rgb_to_hsl,
    rgb_to_hsv,
    rgb_to_lab,
    rgb_to_xyz,
    xyz_to_rgb,
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
    convert_color_space,
    get_color_space,
    list_color_spaces,
    sRGB,
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
    # Color models (from dreamstack.color)
    # ==========================================================================
    "RGBColorModel",
    "HSLColorModel",
    "HSVColorModel",
    "CMYKColorModel",
    # ==========================================================================
    # Manipulation functions (from dreamstack.color)
    # ==========================================================================
    "lighten",
    "darken",
    "saturate",
    "desaturate",
    "grayscale",
    "adjust_hue",
    "mix",
    "invert",
    "fade",
    "brighten",
    "dim",
    "complement",
    # ==========================================================================
    # Harmony functions (from dreamstack.color)
    # ==========================================================================
    "complementary",
    "triadic",
    "tetradic",
    "analogous",
    "split_complementary",
    "square",
    "monochromatic",
    "shades",
    "tints",
    "tones",
    # ==========================================================================
    # Distance & Comparison (from dreamstack.color)
    # ==========================================================================
    "euclidean_distance",
    "manhattan_distance",
    "is_similar",
    "is_dark",
    "is_light",
    "contrast_ratio",
    "passes_wcag_aa",
    "passes_wcag_aaa",
    "luminance",
    # ==========================================================================
    # Gradients (from dreamstack.color)
    # ==========================================================================
    "linear_gradient",
    "bezier_gradient",
    "interpolate",
    # ==========================================================================
    # Generators (from dreamstack.color)
    # ==========================================================================
    "random_color",
    "random_palette",
    "golden_ratio_color",
    # ==========================================================================
    # Validators (from dreamstack.color)
    # ==========================================================================
    "is_valid_hex",
    "is_valid_rgb",
    "is_valid_css_color",
    # ==========================================================================
    # Utils (from dreamstack.color)
    # ==========================================================================
    "NAMED_COLORS",
    "get_named_color",
    "hex_to_rgb",
    "rgb_to_hex",
    "parse_css_color",
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
]
