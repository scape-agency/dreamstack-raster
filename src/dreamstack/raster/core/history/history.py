# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - History
===========================

Undo/redo history manager.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from dreamstack.raster.core.history.history_action import HistoryAction
from dreamstack.raster.core.history.history_state import HistoryState

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.document import Document


class History:
    """
    Manages undo/redo history for a document.

    The history system supports:
    - Unlimited undo/redo (configurable)
    - History state snapshots
    - Action coalescing for related operations
    - History branching

    Example:
        >>> history = History(max_states=100)
        >>> history.push_state("Draw Stroke", document.serialize())
        >>> history.undo(document)
        >>> history.redo(document)
    """

    def __init__(self, max_states: int = 100):
        """
        Initialize history.

        Args:
            max_states: Maximum number of states to keep
        """
        self._max_states = max_states
        self._states: list[HistoryState] = []
        self._current_index = -1
        self._saved_index = -1
        self._action_group: list[HistoryAction] = []
        self._group_name: str = ""
        self._grouping = False
        self._disabled = False
        self._on_change: list[Callable[[], None]] = []

    @property
    def can_undo(self) -> bool:
        """Check if undo is available."""
        return self._current_index > 0

    @property
    def can_redo(self) -> bool:
        """Check if redo is available."""
        return self._current_index < len(self._states) - 1

    @property
    def current_index(self) -> int:
        """Get current history index."""
        return self._current_index

    @property
    def state_count(self) -> int:
        """Get number of states."""
        return len(self._states)

    @property
    def is_modified(self) -> bool:
        """Check if document is modified since last save."""
        return self._current_index != self._saved_index

    @property
    def current_state(self) -> HistoryState | None:
        """Get current state."""
        if 0 <= self._current_index < len(self._states):
            return self._states[self._current_index]
        return None

    @property
    def states(self) -> list[HistoryState]:
        """Get all history states."""
        return self._states.copy()

    def push_state(self, name: str, data: Any) -> None:
        """
        Push a new state onto the history.

        Args:
            name: Action name
            data: State data
        """
        if self._disabled:
            return

        state = HistoryState.create(name, data)

        # Remove any redo states
        if self._current_index < len(self._states) - 1:
            self._states = self._states[: self._current_index + 1]

        # Add new state
        self._states.append(state)
        self._current_index = len(self._states) - 1

        # Trim old states if needed
        while len(self._states) > self._max_states:
            self._states.pop(0)
            self._current_index -= 1
            self._saved_index -= 1

        self._notify_change()

    def undo(self, document: Document) -> bool:
        """
        Undo the last action.

        Args:
            document: Document to restore state to

        Returns:
            True if undo was performed
        """
        if not self.can_undo:
            return False

        self._current_index -= 1
        state = self._states[self._current_index]
        document.deserialize(state.data)

        self._notify_change()
        return True

    def redo(self, document: Document) -> bool:
        """
        Redo the next action.

        Args:
            document: Document to restore state to

        Returns:
            True if redo was performed
        """
        if not self.can_redo:
            return False

        self._current_index += 1
        state = self._states[self._current_index]
        document.deserialize(state.data)

        self._notify_change()
        return True

    def goto(self, index: int, document: Document) -> bool:
        """
        Go to a specific history state.

        Args:
            index: Target state index
            document: Document to restore state to

        Returns:
            True if navigation was successful
        """
        if not 0 <= index < len(self._states):
            return False

        self._current_index = index
        state = self._states[index]
        document.deserialize(state.data)

        self._notify_change()
        return True

    def mark_saved(self) -> None:
        """Mark current state as saved."""
        self._saved_index = self._current_index

    def clear(self) -> None:
        """Clear all history."""
        self._states.clear()
        self._current_index = -1
        self._saved_index = -1
        self._notify_change()

    def begin_group(self, name: str) -> None:
        """
        Begin an action group.

        Multiple actions within a group are treated as a single undo step.

        Args:
            name: Group name
        """
        self._grouping = True
        self._group_name = name
        self._action_group.clear()

    def end_group(self, document: Document) -> None:
        """
        End an action group.

        Args:
            document: Document to snapshot
        """
        if self._grouping:
            self._grouping = False
            if self._action_group:
                self.push_state(self._group_name, document.serialize())
            self._action_group.clear()

    def disable(self) -> None:
        """Disable history recording."""
        self._disabled = True

    def enable(self) -> None:
        """Enable history recording."""
        self._disabled = False

    def on_change(self, callback: Callable[[], None]) -> None:
        """
        Register a change callback.

        Args:
            callback: Function to call when history changes
        """
        self._on_change.append(callback)

    def _notify_change(self) -> None:
        """Notify all change listeners."""
        for callback in self._on_change:
            callback()

    def get_undo_name(self) -> str | None:
        """Get name of action that would be undone."""
        if self.can_undo:
            return self._states[self._current_index].name
        return None

    def get_redo_name(self) -> str | None:
        """Get name of action that would be redone."""
        if self.can_redo:
            return self._states[self._current_index + 1].name
        return None
