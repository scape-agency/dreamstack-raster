"""Rendering intent enumeration."""

from __future__ import annotations

from enum import Enum


class RenderingIntent(Enum):
    """ICC rendering intents."""

    PERCEPTUAL = 0
    RELATIVE_COLORIMETRIC = 1
    SATURATION = 2
    ABSOLUTE_COLORIMETRIC = 3
