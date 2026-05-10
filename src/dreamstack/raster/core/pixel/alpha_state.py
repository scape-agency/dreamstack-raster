# -*- coding: utf-8 -*-

# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Alpha State
================================

Tracks whether the alpha channel is straight (associated/non-premultiplied)
or premultiplied into the color channels.
"""

# =============================================================================
# Imports
# =============================================================================

from __future__ import annotations

from enum import Enum, auto


class AlphaState(Enum):
    """Alpha-association state of an RGBA-like pixel array."""

    STRAIGHT = auto()
    """RGB channels are independent of alpha (a.k.a. unassociated /
    non-premultiplied). The default for the public API."""

    PREMULTIPLIED = auto()
    """RGB channels are premultiplied by alpha. The required state inside
    compositing / blend kernels."""
