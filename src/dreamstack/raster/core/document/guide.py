# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Guide
=========================

Guide line for alignment.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Guide:
    """
    A guide line for alignment.

    Attributes:
        position: Position in pixels
        orientation: 'horizontal' or 'vertical'
        color: Guide color (RGBA)
    """

    position: float
    orientation: str = "horizontal"  # horizontal or vertical
    color: tuple = (0, 255, 255, 255)
