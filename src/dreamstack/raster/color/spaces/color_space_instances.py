# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Pre-defined color space instances."""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

import numpy as np

from dreamstack.raster.color.spaces.color_space import ColorSpace
from dreamstack.raster.color.spaces.gamma_type import GammaType

# Standard white points

# D65 white point (sRGB, Rec.709, Rec.2020, Display P3)
D65 = np.array([0.3127, 0.3290])

# D50 white point (ProPhoto, printing)
D50 = np.array([0.3457, 0.3585])

# ACES white point
ACES_WP = np.array([0.32168, 0.33767])


# Standard color spaces

sRGB = ColorSpace(
    name="sRGB",
    primaries=np.array(
        [
            [0.64, 0.33],  # Red
            [0.30, 0.60],  # Green
            [0.15, 0.06],  # Blue
        ]
    ),
    white_point=D65,
    gamma_type=GammaType.SRGB,
    description="Standard RGB color space for displays",
)

AdobeRGB = ColorSpace(
    name="Adobe RGB (1998)",
    primaries=np.array(
        [
            [0.64, 0.33],
            [0.21, 0.71],
            [0.15, 0.06],
        ]
    ),
    white_point=D65,
    gamma_type=GammaType.POWER,
    gamma=2.2,
    description="Adobe RGB (1998) color space",
)

ProPhotoRGB = ColorSpace(
    name="ProPhoto RGB",
    primaries=np.array(
        [
            [0.7347, 0.2653],
            [0.1596, 0.8404],
            [0.0366, 0.0001],
        ]
    ),
    white_point=D50,
    gamma_type=GammaType.POWER,
    gamma=1.8,
    description="Wide gamut color space for photography",
)

DisplayP3 = ColorSpace(
    name="Display P3",
    primaries=np.array(
        [
            [0.680, 0.320],
            [0.265, 0.690],
            [0.150, 0.060],
        ]
    ),
    white_point=D65,
    gamma_type=GammaType.SRGB,
    description="Apple Display P3 color space",
)

Rec709 = ColorSpace(
    name="Rec. 709",
    primaries=np.array(
        [
            [0.64, 0.33],
            [0.30, 0.60],
            [0.15, 0.06],
        ]
    ),
    white_point=D65,
    gamma_type=GammaType.POWER,
    gamma=2.4,  # BT.1886
    description="ITU-R BT.709 HDTV color space",
)

Rec2020 = ColorSpace(
    name="Rec. 2020",
    primaries=np.array(
        [
            [0.708, 0.292],
            [0.170, 0.797],
            [0.131, 0.046],
        ]
    ),
    white_point=D65,
    gamma_type=GammaType.POWER,
    gamma=2.4,
    description="ITU-R BT.2020 UHDTV color space",
)

ACES = ColorSpace(
    name="ACES",
    primaries=np.array(
        [
            [0.7347, 0.2653],
            [0.0000, 1.0000],
            [0.0001, -0.0770],
        ]
    ),
    white_point=ACES_WP,
    gamma_type=GammaType.LINEAR,
    description="Academy Color Encoding System",
)

ACEScg = ColorSpace(
    name="ACEScg",
    primaries=np.array(
        [
            [0.713, 0.293],
            [0.165, 0.830],
            [0.128, 0.044],
        ]
    ),
    white_point=ACES_WP,
    gamma_type=GammaType.LINEAR,
    description="ACES Computer Graphics color space",
)

DCI_P3 = ColorSpace(
    name="DCI-P3",
    primaries=np.array(
        [
            [0.680, 0.320],
            [0.265, 0.690],
            [0.150, 0.060],
        ]
    ),
    white_point=np.array([0.314, 0.351]),  # DCI white
    gamma_type=GammaType.POWER,
    gamma=2.6,
    description="DCI-P3 digital cinema color space",
)
