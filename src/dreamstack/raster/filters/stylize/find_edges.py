# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Find Edges (Stylize)
========================================

Stylize find edges (inverted edge detection) implementation.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from typing import TYPE_CHECKING

# =============================================================================
# Type Checking Imports
# =============================================================================

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.image import Image


def find_edges(image: Image) -> Image:
    """
    Apply stylize find edges (inverted edge detection).

    Args:
        image: Input image

    Returns:
        Edge-detected image
    """
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.filters.edge.find_edges import (
        find_edges as edge_find_edges,
    )

    return edge_find_edges(image)
