#!/usr/bin/env python
"""Test color module imports."""

import sys

sys.path.insert(0, "src")

# Test convert submodule
from dreamstack.raster.color.convert import (
    convert_color,
    hsv_to_rgb,
    lab_to_rgb,
    rgb_to_hsv,
    rgb_to_lab,
)

print("convert submodule: OK")

# Test spaces submodule
from dreamstack.raster.color.spaces import (
    AdobeRGB,
    ColorSpace,
    GammaType,
    get_color_space,
    sRGB,
)

print("spaces submodule: OK")

# Test profiles submodule
from dreamstack.raster.color.profiles import (
    ICCProfile,
    RenderingIntent,
    load_profile,
)

print("profiles submodule: OK")

# Test palette submodule
from dreamstack.raster.color.palette import (
    BLACK,
    WHITE,
    Color,
    Palette,
    create_gradient,
)

print("palette submodule: OK")

# Test main __init__ imports all correctly
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

print("main __init__: OK")

print()
print("All imports successful!")
