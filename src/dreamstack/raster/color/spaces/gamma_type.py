# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Gamma type enumeration."""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from enum import Enum


class GammaType(Enum):
    """Gamma curve types."""

    LINEAR = "linear"
    SRGB = "srgb"
    POWER = "power"
    LOG = "log"
    PQ = "pq"  # Perceptual Quantizer (HDR)
    HLG = "hlg"  # Hybrid Log-Gamma (HDR)
