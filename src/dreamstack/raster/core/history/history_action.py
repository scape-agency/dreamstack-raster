# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - History Action
==================================

Abstract base for undoable actions.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

# =============================================================================
# Type Checking Imports
# =============================================================================

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.document import Document


class HistoryAction(ABC):
    """
    Abstract base for undoable actions.

    Actions implement the command pattern for reversible operations.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Get action name."""
        ...  # pylint: disable=unnecessary-ellipsis

    @abstractmethod
    def execute(self, document: Document) -> None:
        """Execute the action."""
        ...  # pylint: disable=unnecessary-ellipsis

    @abstractmethod
    def undo(self, document: Document) -> None:
        """Undo the action."""
        ...  # pylint: disable=unnecessary-ellipsis

    def redo(self, document: Document) -> None:
        """Redo the action (default: re-execute)."""
        self.execute(document)
