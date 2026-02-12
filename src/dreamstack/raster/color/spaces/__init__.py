"""
Dreamstack Raster - Color Spaces
================================

Color space definitions and profiles.

"""

from __future__ import annotations

from dreamstack.raster.color.spaces.color_space import ColorSpace
from dreamstack.raster.color.spaces.color_space_instances import (
    ACES,
    ACES_WP,
    D50,
    D65,
    DCI_P3,
    ACEScg,
    AdobeRGB,
    DisplayP3,
    ProPhotoRGB,
    Rec709,
    Rec2020,
    sRGB,
)
from dreamstack.raster.color.spaces.convert_color_space import (
    convert_color_space,
)
from dreamstack.raster.color.spaces.gamma_type import GammaType
from dreamstack.raster.color.spaces.get_color_space import get_color_space
from dreamstack.raster.color.spaces.list_color_spaces import list_color_spaces
from dreamstack.raster.color.spaces.transfer_functions import (
    _hlg_eotf,
    _hlg_oetf,
    _pq_eotf,
    _pq_oetf,
)
from dreamstack.raster.color.spaces.xyz_matrix import (
    _compute_rgb_to_xyz_matrix,
)

__all__: list[str] = [
    # Enum
    "GammaType",
    # Class
    "ColorSpace",
    # Constants
    "D65",
    "D50",
    "ACES_WP",
    # Color space instances
    "sRGB",
    "AdobeRGB",
    "ProPhotoRGB",
    "DisplayP3",
    "Rec709",
    "Rec2020",
    "ACES",
    "ACEScg",
    "DCI_P3",
    # Functions
    "get_color_space",
    "list_color_spaces",
    "convert_color_space",
    # Helper functions (internal but exported)
    "_compute_rgb_to_xyz_matrix",
    "_pq_eotf",
    "_pq_oetf",
    "_hlg_eotf",
    "_hlg_oetf",
]
