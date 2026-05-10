# -*- coding: utf-8 -*-

# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Color Pipeline
===================================

Strict, color-managed pipeline transformations on ``PixelData``.

Every adjustment, filter, or compositing kernel that depends on color
meaning routes its input through :func:`ensure` (or one of the focused
``to_*`` helpers) so it can declare exactly the gamma / working-space /
alpha / dtype state it requires.

The helpers in this module *never* mutate their input — they always
return a new ``PixelData``. They are tolerant of pixel formats whose
encoding semantics (gamma, working space) do not apply (e.g. ``LAB``,
``HSV``, ``CMYK``, single-channel ``GRAY``); for those formats the
gamma / working-space transitions short-circuit to a copy.
"""

# =============================================================================
# Imports
# =============================================================================

from __future__ import annotations

from dreamstack.raster.color.pipeline.ensure import ensure
from dreamstack.raster.color.pipeline.to_alpha import (
    to_premultiplied,
    to_straight,
)
from dreamstack.raster.color.pipeline.to_gamma import to_encoded, to_linear
from dreamstack.raster.color.pipeline.to_working_space import to_working_space

__all__: list[str] = [
    "ensure",
    "to_linear",
    "to_encoded",
    "to_premultiplied",
    "to_straight",
    "to_working_space",
]
