# -*- coding: utf-8 -*-

"""Gamma type enumeration."""

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
