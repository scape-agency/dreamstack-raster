# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - CompressionConfig
=================

Configuration dataclass for smart image compression.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CompressionFormat = Literal["jpeg", "webp", "png"]


@dataclass
class CompressionConfig:
    """Configuration for smart compression.

    Attributes:
        max_size_kb: Maximum file size in kilobytes.
        min_quality: Minimum acceptable quality (1-100).
        max_quality: Starting/maximum quality (1-100).
        format: Output format.
        quality_step: Quality reduction step per iteration.
        max_iterations: Maximum optimization iterations.
    """

    max_size_kb: int = 500
    min_quality: int = 20
    max_quality: int = 95
    format: CompressionFormat = "jpeg"
    quality_step: int = 5
    max_iterations: int = 20
