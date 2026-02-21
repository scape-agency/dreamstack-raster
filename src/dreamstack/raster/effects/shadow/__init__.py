# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Shadow Effects
==============

Drop shadow and inner shadow effects.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from dreamstack.raster.effects.shadow.drop_shadow import drop_shadow
from dreamstack.raster.effects.shadow.inner_shadow import inner_shadow

__all__: list[str] = [
    "drop_shadow",
    "inner_shadow",
]
