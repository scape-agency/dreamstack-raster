# -*- coding: utf-8 -*-

"""
Dreamstack Raster - Analysis Module
===================================

Image analysis: histogram, statistics, color picker.

"""

from __future__ import annotations

from dreamstack.raster.analysis.histogram import (
    cumulative_histogram,
    histogram,
    histogram_luminosity,
    histogram_rgb,
    histogram_stats,
)
from dreamstack.raster.analysis.measure import (
    color_sampler,
    measure_selection,
    pixel_info,
    sample_color,
)
from dreamstack.raster.analysis.statistics import (
    channel_statistics,
    color_count,
    image_statistics,
    unique_colors,
)

__all__: list[str] = [
    # Histogram
    "histogram",
    "histogram_rgb",
    "histogram_luminosity",
    "cumulative_histogram",
    "histogram_stats",
    # Statistics
    "image_statistics",
    "channel_statistics",
    "color_count",
    "unique_colors",
    # Measure
    "measure_selection",
    "pixel_info",
    "sample_color",
    "color_sampler",
]
