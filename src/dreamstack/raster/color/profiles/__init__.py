# -*- coding: utf-8 -*-

"""
Dreamstack Raster - ICC Profile Support
=======================================

ICC color profile handling.

"""

from __future__ import annotations

from dreamstack.raster.color.profiles.color_space_type import ColorSpaceType
from dreamstack.raster.color.profiles.convert_profile import convert_profile
from dreamstack.raster.color.profiles.embed_profile import embed_profile
from dreamstack.raster.color.profiles.get_profile_info import get_profile_info
from dreamstack.raster.color.profiles.get_system_profiles import (
    _get_system_profile_paths,
    get_system_profiles,
)
from dreamstack.raster.color.profiles.icc_profile import ICCProfile
from dreamstack.raster.color.profiles.load_profile import load_profile
from dreamstack.raster.color.profiles.profile_class import ProfileClass
from dreamstack.raster.color.profiles.rendering_intent import RenderingIntent

__all__: list[str] = [
    # Enums
    "RenderingIntent",
    "ProfileClass",
    "ColorSpaceType",
    # Class
    "ICCProfile",
    # Functions
    "load_profile",
    "embed_profile",
    "convert_profile",
    "get_system_profiles",
    "get_profile_info",
    # Internal helpers (exported for compatibility)
    "_get_system_profile_paths",
]
