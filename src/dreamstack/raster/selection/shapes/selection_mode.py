"""
Selection Mode Enum
===================

Defines selection combination modes.

"""

# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from enum import Enum, auto


class SelectionMode(Enum):
    """Mode for combining selections.

    Attributes:
        NEW: Replace existing selection.
        ADD: Add to existing selection.
        SUBTRACT: Subtract from existing selection.
        INTERSECT: Keep only intersection with existing selection.
    """

    NEW = auto()
    ADD = auto()
    SUBTRACT = auto()
    INTERSECT = auto()
