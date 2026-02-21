"""
Invert Selection
================

Invert the selection mask.

"""

# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from dreamstack.raster.selection.shapes.selection import Selection


def invert(selection: Selection) -> Selection:
    """Invert the selection.

    Converts selected areas to unselected and vice versa.

    Args:
        selection: Input selection.

    Returns:
        Inverted selection.

    Example:
        >>> inverted = invert(selection)
    """
    inverted_mask = 255 - selection.mask
    return Selection(mask=inverted_mask)
