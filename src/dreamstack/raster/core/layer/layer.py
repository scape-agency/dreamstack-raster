# -*- coding: utf-8 -*-

"""
Dreamstack Raster - Layer
=========================

Standard raster layer containing pixel data.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from dreamstack.raster.core.bounds import Bounds, Size
from dreamstack.raster.core.layer.blend_mode import BlendMode
from dreamstack.raster.core.layer.layer_base import LayerBase
from dreamstack.raster.core.pixel import BitDepth, PixelData, PixelFormat

if TYPE_CHECKING:
    from dreamstack.raster.core.image import Image


class Layer(LayerBase):
    """
    Standard raster layer containing pixel data.

    This is the most common layer type, holding actual pixel content
    that can be painted, transformed, and manipulated.
    """

    def __init__(
        self,
        pixel_data: PixelData,
        name: str = "Layer",
        opacity: float = 1.0,
        blend_mode: BlendMode = BlendMode.NORMAL,
        visible: bool = True,
        locked: bool = False,
    ):
        """
        Initialize a raster layer.

        Args:
            pixel_data: Layer pixel data
            name: Layer name
            opacity: Layer opacity
            blend_mode: Blend mode
            visible: Visibility
            locked: Lock state
        """
        super().__init__(name, opacity, blend_mode, visible, locked)
        self._pixel_data = pixel_data

    @property
    def pixel_data(self) -> PixelData:
        """Get layer pixel data."""
        return self._pixel_data

    @property
    def width(self) -> int:
        """Get layer width."""
        return self._pixel_data.width

    @property
    def height(self) -> int:
        """Get layer height."""
        return self._pixel_data.height

    @property
    def bounds(self) -> Bounds:
        """Get layer bounds including offset."""
        return Bounds(self._offset.x, self._offset.y, self.width, self.height)

    def render(self, canvas_size: Size) -> NDArray:
        """
        Render layer to canvas size.

        Args:
            canvas_size: Target canvas size

        Returns:
            Rendered RGBA float array
        """
        if not self._visible:
            return np.zeros(
                (int(canvas_size.height), int(canvas_size.width), 4),
                dtype=np.float32,
            )

        # Convert to RGBA float
        if self._pixel_data.pixel_format != PixelFormat.RGBA:
            rgba = self._pixel_data.to_format(PixelFormat.RGBA)
        else:
            rgba = self._pixel_data

        normalized = rgba.to_normalized()
        layer_data = normalized.data

        # Create canvas-sized output
        canvas_h = int(canvas_size.height)
        canvas_w = int(canvas_size.width)
        result = np.zeros((canvas_h, canvas_w, 4), dtype=np.float32)

        # Calculate copy region
        ox, oy = int(self._offset.x), int(self._offset.y)

        # Source bounds
        src_x1 = max(0, -ox)
        src_y1 = max(0, -oy)
        src_x2 = min(self.width, canvas_w - ox)
        src_y2 = min(self.height, canvas_h - oy)

        # Destination bounds
        dst_x1 = max(0, ox)
        dst_y1 = max(0, oy)
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)

        if src_x2 > src_x1 and src_y2 > src_y1:
            result[dst_y1:dst_y2, dst_x1:dst_x2] = layer_data[
                src_y1:src_y2, src_x1:src_x2
            ]

        # Apply mask
        result = self.apply_mask(result)

        # Apply opacity
        result[:, :, 3] *= self._opacity

        return result

    def copy(self) -> Layer:
        """Create a copy of this layer."""
        new_layer = Layer(
            pixel_data=self._pixel_data.copy(),
            name=f"{self._name} copy",
            opacity=self._opacity,
            blend_mode=self._blend_mode,
            visible=self._visible,
            locked=self._locked,
        )
        new_layer._offset = self._offset
        if self._mask is not None:
            new_layer._mask = self._mask.copy()
        return new_layer

    def get_pixel(self, x: int, y: int) -> NDArray:
        """Get pixel value at coordinates."""
        return self._pixel_data.get_pixel(x, y)

    def set_pixel(self, x: int, y: int, value: NDArray | list | tuple) -> None:
        """Set pixel value at coordinates."""
        if self._locked:
            raise RuntimeError("Layer is locked")
        self._pixel_data.set_pixel(x, y, value)

    def fill(self, value: NDArray | list | tuple) -> None:
        """Fill layer with color."""
        if self._locked:
            raise RuntimeError("Layer is locked")
        self._pixel_data.fill(value)

    @classmethod
    def create(
        cls,
        width: int,
        height: int,
        pixel_format: PixelFormat = PixelFormat.RGBA,
        bit_depth: BitDepth = BitDepth.UINT8,
        fill_color: tuple | None = None,
        name: str = "Layer",
    ) -> Layer:
        """
        Create a new blank layer.

        Args:
            width: Layer width
            height: Layer height
            pixel_format: Pixel format
            bit_depth: Bit depth
            fill_color: Optional fill color
            name: Layer name

        Returns:
            New Layer instance
        """
        pixel_data = PixelData.create(
            width=width,
            height=height,
            pixel_format=pixel_format,
            bit_depth=bit_depth,
            fill_value=fill_color,
        )
        return cls(pixel_data, name=name)

    @classmethod
    def from_image(cls, image: Image, name: str | None = None) -> Layer:
        """
        Create layer from image.

        Args:
            image: Source image
            name: Optional layer name

        Returns:
            New Layer instance
        """
        return cls(pixel_data=image.pixel_data.copy(), name=name or image.name)
