"""
Dreamstack Raster - History State
=================================

Single state representation in history.

"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class HistoryState:
    """
    Represents a single state in the history.

    Attributes:
        name: Human-readable name of the action
        timestamp: When the action occurred
        data: Serialized state data
        thumbnail: Optional preview thumbnail
    """

    name: str
    timestamp: datetime
    data: Any
    thumbnail: bytes | None = None

    @classmethod
    def create(cls, name: str, data: Any) -> HistoryState:
        """
        Create a new history state.

        Args:
            name: Action name
            data: State data

        Returns:
            New HistoryState
        """
        return cls(name=name, timestamp=datetime.now(), data=data)
