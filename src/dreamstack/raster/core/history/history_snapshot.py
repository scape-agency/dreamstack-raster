# -*- coding: utf-8 -*-

"""
Dreamstack Raster - History Snapshot
====================================

Lightweight snapshot for quick state capture.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dreamstack.raster.core.document import Document


class HistorySnapshot:
    """
    A lightweight snapshot for quick state capture.

    Used for operations that need temporary state storage
    without full serialization.
    """

    def __init__(self, document: Document):
        """
        Create snapshot from document.

        Args:
            document: Document to snapshot
        """
        self._layer_data = {}
        self._selection = None

        # Snapshot layer pixel data
        for layer in document.layers.flatten_hierarchy():
            if hasattr(layer, "pixel_data"):
                self._layer_data[layer.id] = layer.pixel_data.data.copy()

        # Snapshot selection
        if document.selection is not None:
            self._selection = document.selection.mask.copy()

    def restore(self, document: Document) -> None:
        """
        Restore snapshot to document.

        Args:
            document: Document to restore to
        """
        for layer in document.layers.flatten_hierarchy():
            if layer.id in self._layer_data and hasattr(layer, "pixel_data"):
                layer.pixel_data.data[:] = self._layer_data[layer.id]

        if self._selection is not None and document.selection is not None:
            document.selection.mask[:] = self._selection
