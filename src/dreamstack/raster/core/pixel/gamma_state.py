# -*- coding: utf-8 -*-

# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Gamma State
================================

Tracks whether pixel values are linear-light or display-encoded
(e.g. sRGB-encoded). The companion ``ColorSpace`` provides the actual
transfer function used to move between the two states.
"""

# =============================================================================
# Imports
# =============================================================================

from __future__ import annotations

from enum import Enum, auto


class GammaState(Enum):
    """Whether pixel values are linear-light or display-encoded."""

    LINEAR = auto()
    """Linear-light values (suitable for compositing/blending math)."""

    ENCODED = auto()
    """Display-encoded values via the working space's transfer function
    (e.g. sRGB-encoded). The default for I/O and most adjustments."""
