# -*- coding: utf-8 -*-

"""
Dreamstack Raster - Layer Base
==============================

Abstract base class for all layer types.

"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import numpy as np
from numpy.typing import NDArray

from dreamstack.raster.core.bounds import Bounds, Point, Size
from dreamstack.raster.core.layer.blend_mode import BlendMode

if TYPE_CHECKING:
    from dreamstack.raster.core.layer.layer_group import LayerGroup


class LayerBase(ABC):
    """
    Abstract base class for all layer types.
    """

    def __init__(
        self,
        name: str = "Layer",
        opacity: float = 1.0,
        blend_mode: BlendMode = BlendMode.NORMAL,
        visible: bool = True,
        locked: bool = False,
    ):
        """
        Initialize layer base.

        Args:
            name: Layer name
            opacity: Layer opacity (0-1)
            blend_mode: Blend mode
            visible: Whether layer is visible
            locked: Whether layer is locked
        """
        self._id = str(uuid.uuid4())
        self._name = name
        self._opacity = opacity
        self._blend_mode = blend_mode
        self._visible = visible
        self._locked = locked
        self._parent: Optional[LayerGroup] = None
        self._mask: Optional[NDArray] = None
        self._mask_enabled = True
        self._offset = Point(0, 0)

    @property
    def id(self) -> str:
        """Get unique layer ID."""
        return self._id

    @property
    def name(self) -> str:
        """Get layer name."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """Set layer name."""
        self._name = value

    @property
    def opacity(self) -> float:
        """Get layer opacity."""
        return self._opacity

    @opacity.setter
    def opacity(self, value: float) -> None:
        """Set layer opacity (0-1)."""
        self._opacity = max(0.0, min(1.0, value))

    @property
    def blend_mode(self) -> BlendMode:
        """Get blend mode."""
        return self._blend_mode

    @blend_mode.setter
    def blend_mode(self, value: BlendMode) -> None:
        """Set blend mode."""
        self._blend_mode = value

    @property
    def visible(self) -> bool:
        """Check if layer is visible."""
        return self._visible

    @visible.setter
    def visible(self, value: bool) -> None:
        """Set layer visibility."""
        self._visible = value

    @property
    def locked(self) -> bool:
        """Check if layer is locked."""
        return self._locked

    @locked.setter
    def locked(self, value: bool) -> None:
        """Set layer lock state."""
        self._locked = value

    @property
    def parent(self) -> Optional[LayerGroup]:
        """Get parent layer group."""
        return self._parent

    @property
    def mask(self) -> Optional[NDArray]:
        """Get layer mask."""
        return self._mask

    @mask.setter
    def mask(self, value: Optional[NDArray]) -> None:
        """Set layer mask."""
        self._mask = value

    @property
    def mask_enabled(self) -> bool:
        """Check if mask is enabled."""
        return self._mask_enabled

    @mask_enabled.setter
    def mask_enabled(self, value: bool) -> None:
        """Set mask enabled state."""
        self._mask_enabled = value

    @property
    def offset(self) -> Point:
        """Get layer offset."""
        return self._offset

    @offset.setter
    def offset(self, value: Point) -> None:
        """Set layer offset."""
        self._offset = value

    @property
    @abstractmethod
    def bounds(self) -> Bounds:
        """Get layer bounds."""
        pass

    @abstractmethod
    def render(self, canvas_size: Size) -> NDArray:
        """
        Render layer content to array.

        Args:
            canvas_size: Size of canvas to render to

        Returns:
            Rendered pixel data as normalized float array
        """
        pass

    @abstractmethod
    def copy(self) -> LayerBase:
        """Create a copy of this layer."""
        pass

    def apply_mask(self, data: NDArray) -> NDArray:
        """
        Apply layer mask to pixel data.

        Args:
            data: Pixel data (must have alpha channel)

        Returns:
            Data with mask applied to alpha
        """
        if self._mask is None or not self._mask_enabled:
            return data

        result = data.copy()

        # Normalize mask
        if np.issubdtype(self._mask.dtype, np.integer):
            mask = (
                self._mask.astype(np.float32) / np.iinfo(self._mask.dtype).max
            )
        else:
            mask = self._mask.astype(np.float32)

        # Apply mask to alpha channel
        if result.shape[2] >= 4:
            result[:, :, 3] *= mask

        return result
