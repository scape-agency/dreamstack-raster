"""Resize method enumeration."""

# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from enum import StrEnum


class ResizeMethod(StrEnum):
    """Resize fitting method.

    Attributes:
        STRETCH: Stretch to exact dimensions (distorts aspect ratio).
        FIT: Scale to fit within dimensions (maintains aspect).
        FILL: Scale to fill dimensions, cropping if needed.
        PAD: Scale to fit and pad to exact dimensions.
    """

    STRETCH = "stretch"
    FIT = "fit"
    FILL = "fill"
    PAD = "pad"
