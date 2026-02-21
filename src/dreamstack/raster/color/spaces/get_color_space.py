# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Get color space by name."""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from dreamstack.raster.color.spaces.color_space import ColorSpace
from dreamstack.raster.color.spaces.color_space_instances import (
    ACES,
    DCI_P3,
    ACEScg,
    AdobeRGB,
    DisplayP3,
    ProPhotoRGB,
    Rec709,
    Rec2020,
    sRGB,
)

# Color space registry
_color_spaces = {
    "srgb": sRGB,
    "adobe": AdobeRGB,
    "adobe rgb": AdobeRGB,
    "adobergb": AdobeRGB,
    "prophoto": ProPhotoRGB,
    "prophoto rgb": ProPhotoRGB,
    "p3": DisplayP3,
    "display p3": DisplayP3,
    "displayp3": DisplayP3,
    "rec709": Rec709,
    "rec.709": Rec709,
    "bt709": Rec709,
    "rec2020": Rec2020,
    "rec.2020": Rec2020,
    "bt2020": Rec2020,
    "aces": ACES,
    "acescg": ACEScg,
    "dci-p3": DCI_P3,
    "dci p3": DCI_P3,
}


def get_color_space(name: str) -> ColorSpace | None:
    """
    Get a color space by name.

    Args:
        name: Color space name (case-insensitive)

    Returns:
        ColorSpace or None if not found
    """
    return _color_spaces.get(name.lower())
