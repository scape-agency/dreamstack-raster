# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Adjustment Layer
====================================

Non-destructive adjustment layer.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dreamstack.raster.core.bounds import Bounds, Size
from dreamstack.raster.core.layer.blend_mode import BlendMode
from dreamstack.raster.core.layer.layer_base import LayerBase


class AdjustmentLayer(LayerBase):
    """
    Non-destructive adjustment layer.

    Adjustment layers apply color corrections and adjustments
    to all layers below them without modifying the original pixels.
    """

    def __init__(
        self,
        adjustment_type: str,
        parameters: dict,
        name: str = "Adjustment",
        opacity: float = 1.0,
        blend_mode: BlendMode = BlendMode.NORMAL,
        visible: bool = True,
    ):
        """
        Initialize an adjustment layer.

        Args:
            adjustment_type: Type of adjustment
            parameters: Adjustment parameters
            name: Layer name
            opacity: Layer opacity
            blend_mode: Blend mode
            visible: Visibility
        """
        super().__init__(name, opacity, blend_mode, visible)
        self._adjustment_type = adjustment_type
        self._parameters = parameters
        self._apply_func: Callable | None = None

    @property
    def adjustment_type(self) -> str:
        """Get adjustment type."""
        return self._adjustment_type

    @property
    def parameters(self) -> dict:
        """Get adjustment parameters."""
        return self._parameters.copy()

    def set_parameter(self, key: str, value: Any) -> None:
        """
        Set an adjustment parameter.

        Args:
            key: Parameter name
            value: Parameter value
        """
        self._parameters[key] = value

    @property
    def bounds(self) -> Bounds:
        """Adjustment layers cover the entire canvas."""
        return Bounds(0, 0, float("inf"), float("inf"))

    def apply_to(self, data: NDArray) -> NDArray:
        """
        Apply adjustment to pixel data.

        Args:
            data: Input pixel data (normalized float)

        Returns:
            Adjusted pixel data
        """
        from dreamstack.raster.adjustments import apply_adjustment

        result = apply_adjustment(
            data, self._adjustment_type, self._parameters
        )

        # Apply mask if present
        if self._mask is not None and self._mask_enabled:
            if np.issubdtype(self._mask.dtype, np.integer):
                mask = (
                    self._mask.astype(np.float32)
                    / np.iinfo(self._mask.dtype).max
                )
            else:
                mask = self._mask.astype(np.float32)

            # Blend adjusted with original based on mask
            mask = mask[:, :, np.newaxis]
            result = data * (1 - mask) + result * mask

        return result

    def render(self, canvas_size: Size) -> NDArray:
        """
        Adjustment layers don't render directly.

        They are applied during compositing.
        """
        return np.zeros(
            (int(canvas_size.height), int(canvas_size.width), 4),
            dtype=np.float32,
        )

    def copy(self) -> AdjustmentLayer:
        """Create a copy of this adjustment layer."""
        new_layer = AdjustmentLayer(
            adjustment_type=self._adjustment_type,
            parameters=self._parameters.copy(),
            name=f"{self._name} copy",
            opacity=self._opacity,
            blend_mode=self._blend_mode,
            visible=self._visible,
        )
        if self._mask is not None:
            new_layer._mask = (
                self._mask.copy()
            )  # pylint: disable=protected-access
        return new_layer
