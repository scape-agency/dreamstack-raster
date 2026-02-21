# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Profile class enumeration."""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from enum import Enum


class ProfileClass(Enum):
    """ICC profile classes."""

    INPUT = "scnr"
    DISPLAY = "mntr"
    OUTPUT = "prtr"
    LINK = "link"
    ABSTRACT = "abst"
    COLORSPACE = "spac"
    NAMED_COLOR = "nmcl"
