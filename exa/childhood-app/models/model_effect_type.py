"""
Effect Type
===========

Enum for available effect types.
"""

from __future__ import annotations

from enum import Enum


class EffectType(str, Enum):
    """Available effect types."""

    DROP_SHADOW = "drop_shadow"
    WARM_FILTER = "warm_filter"
    COOL_FILTER = "cool_filter"
    VINTAGE = "vintage"
    HIGH_CONTRAST = "high_contrast"
    SOFT_GLOW = "soft_glow"
    SHARPEN = "sharpen"
    BLUR = "blur"
    SEPIA = "sepia"
