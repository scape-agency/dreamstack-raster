# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Extraction Config
=================

Configuration dataclass for object extraction.
"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from dataclasses import dataclass, field

from dreamstack.raster.analysis.contour.detector import DetectionConfig
from dreamstack.raster.analysis.preprocessing.processor import (
    PreprocessingConfig,
)


@dataclass
class ExtractionConfig:
    """Configuration for object extraction.

    Parameters
    ----------
    margin : int
        Margin around extracted objects. Default 25.
    min_dimension : int
        Minimum width/height for valid objects. Default 24.
    min_area_ratio : float
        Minimum object area as ratio of image. Default 0.0002.
    max_area_ratio : float
        Maximum object area as ratio of image. Default 0.95.
    target_size : int | None
        Target size for output images. None = no resize.
    with_alpha : bool
        Extract with transparent background. Default False.
    feather_edges : int
        Feathering for alpha edges. Default 0.
    preprocessing : PreprocessingConfig
        Preprocessing configuration.
    detection : DetectionConfig
        Contour detection configuration.

    Examples
    --------
    >>> config = ExtractionConfig(margin=50, with_alpha=True, feather_edges=3)
    >>> extractor = ObjectExtractor(config)
    """

    margin: int = 25
    min_dimension: int = 24
    min_area_ratio: float = 0.0002
    max_area_ratio: float = 0.95
    target_size: int | None = None
    with_alpha: bool = False
    feather_edges: int = 0
    preprocessing: PreprocessingConfig = field(
        default_factory=PreprocessingConfig
    )
    detection: DetectionConfig = field(default_factory=DetectionConfig)
