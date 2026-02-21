"""
Dreamstack Raster - Text Layer
==============================

Text layer with editable text content.

"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from dreamstack.raster.core.bounds import Bounds, Size
from dreamstack.raster.core.layer.blend_mode import BlendMode
from dreamstack.raster.core.layer.layer_base import LayerBase
from dreamstack.raster.core.pixel import BitDepth, PixelData, PixelFormat


class TextLayer(LayerBase):
    """
    Text layer with editable text content.

    Text layers maintain editable text that can be re-rendered
    at any time with different fonts, sizes, or styles.
    """

    def __init__(
        self,
        text: str,
        font_family: str = "Arial",
        font_size: float = 24.0,
        font_color: tuple = (0, 0, 0, 255),
        name: str = "Text",
        opacity: float = 1.0,
        blend_mode: BlendMode = BlendMode.NORMAL,
        visible: bool = True,
    ):
        """
        Initialize a text layer.

        Args:
            text: Text content
            font_family: Font family name
            font_size: Font size in points
            font_color: Text color (RGBA)
            name: Layer name
            opacity: Layer opacity
            blend_mode: Blend mode
            visible: Visibility
        """
        super().__init__(name, opacity, blend_mode, visible)
        self._text = text
        self._font_family = font_family
        self._font_size = font_size
        self._font_color = font_color
        self._font_weight = "normal"
        self._font_style = "normal"
        self._line_height = 1.2
        self._alignment = "left"
        self._kerning = 0.0
        self._cached_render: NDArray | None = None
        self._cached_bounds: Bounds | None = None

    @property
    def text(self) -> str:
        """Get text content."""
        return self._text

    @text.setter
    def text(self, value: str) -> None:
        """Set text content."""
        self._text = value
        self._invalidate_cache()

    @property
    def font_family(self) -> str:
        """Get font family."""
        return self._font_family

    @font_family.setter
    def font_family(self, value: str) -> None:
        """Set font family."""
        self._font_family = value
        self._invalidate_cache()

    @property
    def font_size(self) -> float:
        """Get font size."""
        return self._font_size

    @font_size.setter
    def font_size(self, value: float) -> None:
        """Set font size."""
        self._font_size = value
        self._invalidate_cache()

    @property
    def font_color(self) -> tuple:
        """Get font color."""
        return self._font_color

    @font_color.setter
    def font_color(self, value: tuple) -> None:
        """Set font color."""
        self._font_color = value
        self._invalidate_cache()

    @property
    def bounds(self) -> Bounds:
        """Get text bounds."""
        if self._cached_bounds is None:
            self._render_text()
        return self._cached_bounds or Bounds(0, 0, 0, 0)

    def _invalidate_cache(self) -> None:
        """Invalidate cached render."""
        self._cached_render = None
        self._cached_bounds = None

    def _render_text(self) -> None:
        """Render text to pixel array."""
        # pylint: disable=import-outside-toplevel
        from dreamstack.raster.drawing.text import render_text

        self._cached_render, self._cached_bounds = render_text(  # type: ignore[assignment]
            text=self._text,
            font_family=self._font_family,
            font_size=self._font_size,
            font_color=self._font_color,
            font_weight=self._font_weight,
            font_style=self._font_style,
            line_height=self._line_height,
            alignment=self._alignment,
        )

    def render(self, canvas_size: Size) -> NDArray:
        """
        Render text layer.

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

        if self._cached_render is None:
            self._render_text()

        # Create canvas-sized output
        canvas_h = int(canvas_size.height)
        canvas_w = int(canvas_size.width)
        result = np.zeros((canvas_h, canvas_w, 4), dtype=np.float32)

        if self._cached_render is not None:
            text_h, text_w = self._cached_render.shape[:2]
            ox, oy = int(self._offset.x), int(self._offset.y)

            # Calculate copy region
            src_x1 = max(0, -ox)
            src_y1 = max(0, -oy)
            src_x2 = min(text_w, canvas_w - ox)
            src_y2 = min(text_h, canvas_h - oy)

            dst_x1 = max(0, ox)
            dst_y1 = max(0, oy)
            dst_x2 = dst_x1 + (src_x2 - src_x1)
            dst_y2 = dst_y1 + (src_y2 - src_y1)

            if src_x2 > src_x1 and src_y2 > src_y1:
                result[dst_y1:dst_y2, dst_x1:dst_x2] = self._cached_render[
                    src_y1:src_y2, src_x1:src_x2
                ]

        # Apply mask and opacity
        result = self.apply_mask(result)
        result[:, :, 3] *= self._opacity

        return result

    def copy(self) -> TextLayer:
        """Create a copy of this text layer."""
        new_layer = TextLayer(
            text=self._text,
            font_family=self._font_family,
            font_size=self._font_size,
            font_color=self._font_color,
            name=f"{self._name} copy",
            opacity=self._opacity,
            blend_mode=self._blend_mode,
            visible=self._visible,
        )
        # pylint: disable=protected-access
        new_layer._offset = self._offset
        new_layer._font_weight = self._font_weight
        new_layer._font_style = self._font_style
        new_layer._line_height = self._line_height
        new_layer._alignment = self._alignment

        if self._mask is not None:
            new_layer._mask = self._mask.copy()
        # pylint: enable=protected-access

        return new_layer

    def rasterize(self, canvas_size: Size):
        """
        Convert text to raster layer.

        Args:
            canvas_size: Canvas size

        Returns:
            Rasterized Layer
        """
        # pylint: disable=import-outside-toplevel
        from dreamstack.raster.core.layer.layer import Layer

        rendered = self.render(canvas_size)

        # Convert to uint8
        uint8_data = (rendered * 255).clip(0, 255).astype(np.uint8)

        pixel_data = PixelData(
            data=uint8_data,
            pixel_format=PixelFormat.RGBA,
            bit_depth=BitDepth.UINT8,
        )

        return Layer(pixel_data, name=self._name)
