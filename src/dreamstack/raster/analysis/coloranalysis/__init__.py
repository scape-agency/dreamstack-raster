# -*- coding: utf-8 -*-

"""
Dreamstack Raster - Color Analysis Module
=========================================

Color analysis for images including dominant color detection,
k-means clustering, and background color analysis.

"""

from __future__ import annotations

from dreamstack.raster.analysis.coloranalysis.analyzer import ColorAnalyzer
from dreamstack.raster.analysis.coloranalysis.operations import (
    adjust_background_color,
    create_gradient_background,
    find_background_color,
    get_dominant_color,
    get_dominant_colors,
    get_most_common_color,
)

__all__: list[str] = [
    # Analyzer Class
    "ColorAnalyzer",
    # Operations
    "get_dominant_color",
    "get_dominant_colors",
    "get_most_common_color",
    "find_background_color",
    "adjust_background_color",
    "create_gradient_background",
]
