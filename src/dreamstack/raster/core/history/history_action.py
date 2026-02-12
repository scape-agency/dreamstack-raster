"""
Dreamstack Raster - History Action
==================================

Abstract base for undoable actions.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
        pass

    @abstractmethod
    def execute(self, document: Document) -> None:
        """Execute the action."""
        pass

    @abstractmethod
    def undo(self, document: Document) -> None:
        """Undo the action."""
        pass

    def redo(self, document: Document) -> None:
        """Redo the action (default: re-execute)."""
        self.execute(document)
