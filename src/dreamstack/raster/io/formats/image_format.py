# -*- coding: utf-8 -*-

"""
Dreamstack Raster - Image Format Enum
=====================================

Supported image formats enumeration.

"""

from __future__ import annotations

from enum import Enum, auto


class ImageFormat(Enum):
    """Supported image formats."""

    # Raster formats
    PNG = auto()
    JPEG = auto()
    TIFF = auto()
    BMP = auto()
    GIF = auto()
    WEBP = auto()
    ICO = auto()

    # High dynamic range
    EXR = auto()
    HDR = auto()

    # Professional
    PSD = auto()
    PSB = auto()
    XCF = auto()

    # Raw formats
    RAW = auto()
    CR2 = auto()
    CR3 = auto()
    NEF = auto()
    ARW = auto()
    DNG = auto()
    ORF = auto()
    RW2 = auto()

    # Vector (rasterize on load)
    SVG = auto()
    PDF = auto()
    AI = auto()
    EPS = auto()

    # Other
    HEIC = auto()
    HEIF = auto()
    AVIF = auto()
    JXL = auto()
