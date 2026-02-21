# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Color Bridge Module
=======================================

Bridge between dreamstack-raster (numpy arrays) and dreamstack-color (color models).
Provides utilities for converting between vectorized array operations and single-color
operations from the dreamstack-color library.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from dreamstack.raster.color.bridge.array_to_model import (
    array_to_rgb,
    arrays_to_rgb_list,
    rgb_list_to_arrays,
    rgb_to_array,
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
