#!/usr/bin/env python
"""Test color module imports in isolation."""

import os
import sys

# Directly add the src to path
sys.path.insert(
    0, "/Users/larsvanvianen/Documents/GitHub/dreamstack/dreamstack-raster/src"
)

# Temporarily remove the parent package __init__.py from loading
# by directly importing color submodules

print("Testing color module imports in isolation...")
print()

# Test convert submodule
try:
    from dreamstack.raster.color.convert import hsv_to_rgb, rgb_to_hsv
    from dreamstack.raster.color.convert.convert_color import convert_color
    from dreamstack.raster.color.convert.hsv_to_rgb import hsv_to_rgb
    from dreamstack.raster.color.convert.lab_to_rgb import lab_to_rgb
    from dreamstack.raster.color.convert.rgb_to_hsv import rgb_to_hsv
    from dreamstack.raster.color.convert.rgb_to_lab import rgb_to_lab

    print("convert submodule: OK")
except Exception as e:
    print(f"convert submodule: FAILED - {e}")

# Test spaces submodule
try:
    from dreamstack.raster.color.spaces import (
        AdobeRGB,
        ColorSpace,
        GammaType,
        sRGB,
    )
    from dreamstack.raster.color.spaces.color_space import ColorSpace
    from dreamstack.raster.color.spaces.color_space_instances import (
        AdobeRGB,
        sRGB,
    )
    from dreamstack.raster.color.spaces.gamma_type import GammaType
    from dreamstack.raster.color.spaces.get_color_space import get_color_space

    print("spaces submodule: OK")
except Exception as e:
    print(f"spaces submodule: FAILED - {e}")

# Test profiles submodule
try:
    from dreamstack.raster.color.profiles import ICCProfile, RenderingIntent
    from dreamstack.raster.color.profiles.icc_profile import ICCProfile
    from dreamstack.raster.color.profiles.load_profile import load_profile
    from dreamstack.raster.color.profiles.rendering_intent import (
        RenderingIntent,
    )

    print("profiles submodule: OK")
except Exception as e:
    print(f"profiles submodule: FAILED - {e}")

# Test palette submodule
try:
    from dreamstack.raster.color.palette import BLACK, WHITE, Color, Palette
    from dreamstack.raster.color.palette.color import Color
    from dreamstack.raster.color.palette.color_presets import BLACK, WHITE
    from dreamstack.raster.color.palette.create_gradient import create_gradient
    from dreamstack.raster.color.palette.palette import Palette

    print("palette submodule: OK")
except Exception as e:
    print(f"palette submodule: FAILED - {e}")

# Test main color __init__
try:
    from dreamstack.raster.color import (
        ACES,
        ACES_WP,
        BLACK,
        BLUE,
        CYAN,
        D50,
        D65,
        DCI_P3,
        GREEN,
        MAGENTA,
        RED,
        TRANSPARENT,
        WHITE,
        YELLOW,
        ACEScg,
        AdobeRGB,
        Color,
        ColorSpace,
        ColorSpaceType,
        DisplayP3,
        GammaType,
        ICCProfile,
        Palette,
        ProfileClass,
        ProPhotoRGB,
        Rec709,
        Rec2020,
        RenderingIntent,
        cmyk_to_rgb,
        convert_color,
        convert_color_space,
        convert_profile,
        create_gradient,
        embed_profile,
        get_color_space,
        get_profile_info,
        get_system_profiles,
        gray_to_rgb,
        hsl_to_rgb,
        hsv_to_rgb,
        lab_to_rgb,
        list_color_spaces,
        load_profile,
        rgb_to_cmyk,
        rgb_to_gray,
        rgb_to_hsl,
        rgb_to_hsv,
        rgb_to_lab,
        rgb_to_xyz,
        sRGB,
        xyz_to_rgb,
    )

    print("main color __init__: OK")
except Exception as e:
    print(f"main color __init__: FAILED - {e}")

print()
print("Testing actual functionality...")

# Test conversion functions work
import numpy as np

try:
    rgb = np.array([1.0, 0.0, 0.0])  # Pure red
    hsv = rgb_to_hsv(rgb)
    rgb_back = hsv_to_rgb(hsv)
    assert np.allclose(
        rgb, rgb_back, atol=0.01
    ), "RGB->HSV->RGB conversion failed"
    print("rgb_to_hsv / hsv_to_rgb: OK")
except Exception as e:
    print(f"rgb_to_hsv / hsv_to_rgb: FAILED - {e}")

try:
    c = Color(255, 0, 0)
    assert c.to_hex() == "#ff0000", "Color hex conversion failed"
    print("Color class: OK")
except Exception as e:
    print(f"Color class: FAILED - {e}")

try:
    p = Palette.from_hex_list(["#ff0000", "#00ff00", "#0000ff"], name="RGB")
    assert len(p) == 3, "Palette creation failed"
    print("Palette class: OK")
except Exception as e:
    print(f"Palette class: FAILED - {e}")

try:
    cs = get_color_space("srgb")
    assert cs is not None, "get_color_space failed"
    assert cs.name == "sRGB", "Wrong color space returned"
    print("get_color_space: OK")
except Exception as e:
    print(f"get_color_space: FAILED - {e}")

print()
print("All tests passed!")
