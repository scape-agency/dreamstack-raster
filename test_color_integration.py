# -*- coding: utf-8 -*-
# ruff: noqa: F401, F811, E501
# pylint: disable=W0611,W0404
# pyright: reportUnusedImport=false

"""
Test Color Integration
======================

Tests the integration between dreamstack-raster and dreamstack-color.
This test runs standalone without requiring scipy or other heavy dependencies.
"""

import os
import sys

# Set up paths before any imports
raster_src = os.path.abspath("src")
color_src = os.path.abspath("../dreamstack-color/src")
sys.path.insert(0, color_src)  # Color first for priority
sys.path.insert(0, raster_src)

print("Testing imports...")

# Import dreamstack.color first (no heavy dependencies)
# pylint: disable=wrong-import-position,wrong-import-order
from dreamstack.color import complement, darken, lighten

print("✓ dreamstack.color imports successful")

# Import conversion functions directly from their modules
# We need to import these without going through the raster __init__.py
import importlib.util

import numpy as np

# pylint: enable=wrong-import-position,wrong-import-order

# Load rgb_to_hsv module directly
spec = importlib.util.spec_from_file_location(
    "rgb_to_hsv",
    os.path.join(raster_src, "dreamstack/raster/color/convert/rgb_to_hsv.py"),
)
assert spec is not None and spec.loader is not None
rgb_to_hsv_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rgb_to_hsv_module)
array_rgb_to_hsv = rgb_to_hsv_module.rgb_to_hsv

spec = importlib.util.spec_from_file_location(
    "hsv_to_rgb",
    os.path.join(raster_src, "dreamstack/raster/color/convert/hsv_to_rgb.py"),
)
assert spec is not None and spec.loader is not None
hsv_to_rgb_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hsv_to_rgb_module)
array_hsv_to_rgb = hsv_to_rgb_module.hsv_to_rgb

print("✓ Array conversion functions loaded")

# Load bridge module directly
spec = importlib.util.spec_from_file_location(
    "dreamstack.raster.color.bridge.array_to_model",
    os.path.join(
        raster_src, "dreamstack/raster/color/bridge/array_to_model.py"
    ),
)
assert spec is not None and spec.loader is not None
bridge_module = importlib.util.module_from_spec(spec)
sys.modules["dreamstack.raster.color.bridge.array_to_model"] = bridge_module
spec.loader.exec_module(bridge_module)
array_to_rgb = bridge_module.array_to_rgb
rgb_to_array = bridge_module.rgb_to_array

print("✓ Bridge functions loaded")

# Load Color class directly - need to register module for dataclass to work
spec = importlib.util.spec_from_file_location(
    "dreamstack.raster.color.palette.color",
    os.path.join(raster_src, "dreamstack/raster/color/palette/color.py"),
)
assert spec is not None and spec.loader is not None
color_module = importlib.util.module_from_spec(spec)
sys.modules["dreamstack.raster.color.palette.color"] = color_module
spec.loader.exec_module(color_module)
Color = color_module.Color

print("✓ Color class loaded")

print("\n✓ All imports successful")

# Test array-based conversion
import numpy as np  # pylint: disable=wrong-import-position,wrong-import-order

print("\nTesting array-based conversions...")
pixel = np.array([0.5, 0.3, 0.2])
hsv = array_rgb_to_hsv(pixel)
print(f"✓ Array RGB {pixel} -> HSV {hsv}")

back_to_rgb = array_hsv_to_rgb(hsv)
print(f"✓ Array HSV {hsv} -> RGB {back_to_rgb}")

# Test bridge functions
print("\nTesting bridge functions...")
rgb_model = array_to_rgb(pixel, normalized=True)
print(f"✓ array_to_rgb: {rgb_model}")

array_back = rgb_to_array(rgb_model, normalized=True)
print(f"✓ rgb_to_array: {array_back}")

# Test dreamstack.color manipulation
print("\nTesting dreamstack.color manipulation...")
lighter = lighten(rgb_model, 20)
print(f"✓ lighten(rgb, 20): {lighter}")

darker = darken(rgb_model, 20)
print(f"✓ darken(rgb, 20): {darker}")

complemented = complement(rgb_model)
print(f"✓ complement(rgb): {complemented}")

# Test Color class
print("\nTesting Color class...")
color = Color(128, 64, 32)
print(f"✓ Color(128, 64, 32): {color.to_hex()}")

ds_rgb = color.to_dreamstack_rgb()
print(f"✓ to_dreamstack_rgb(): {ds_rgb}")

lighter_color = color.lighten(0.1)
print(f"✓ Color.lighten(0.1): {lighter_color.to_hex()}")

darker_color = color.darken(0.1)
print(f"✓ Color.darken(0.1): {darker_color.to_hex()}")

complement_color = color.complement()
print(f"✓ Color.complement(): {complement_color.to_hex()}")

# Test from_hsv and from_hsl
print("\nTesting Color.from_hsv/from_hsl...")
color_from_hsv = Color.from_hsv(180, 0.5, 0.8)
print(f"✓ Color.from_hsv(180, 0.5, 0.8): {color_from_hsv.to_hex()}")

color_from_hsl = Color.from_hsl(180, 0.5, 0.5)
print(f"✓ Color.from_hsl(180, 0.5, 0.5): {color_from_hsl.to_hex()}")

# Test to_hsv and to_hsl
print("\nTesting Color.to_hsv/to_hsl...")
hsv_tuple = color.to_hsv()
print(f"✓ Color(128, 64, 32).to_hsv(): {hsv_tuple}")

hsl_tuple = color.to_hsl()
print(f"✓ Color(128, 64, 32).to_hsl(): {hsl_tuple}")

print("\n" + "=" * 50)
print("All tests passed! ✓")
print("=" * 50)
