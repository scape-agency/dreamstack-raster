# -*- coding: utf-8 -*-

"""
Dreamstack Raster - Color Bridge Module
=======================================

Bridge between dreamstack-raster (numpy arrays) and dreamstack-color (color models).
Provides utilities for converting between vectorized array operations and single-color
operations from the dreamstack-color library.

"""

from __future__ import annotations

from dreamstack.raster.color.bridge.array_to_model import (
    array_to_rgb,
    arrays_to_rgb_list,
    rgb_to_array,
    rgb_list_to_arrays,
)
from dreamstack.raster.color.bridge.model_conversions import (
    convert_color_model,
    get_color_model,
)

__all__: list[str] = [
    # Array <-> Model conversions
    "array_to_rgb",
    "rgb_to_array",
    "arrays_to_rgb_list",
    "rgb_list_to_arrays",
    # Model utilities
    "convert_color_model",
    "get_color_model",
]
