"""
Dreamstack Raster - Canvas
==========================

Canvas for rendering and compositing layers.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from dreamstack.raster.core.bounds import Bounds, Point, Size
from dreamstack.raster.core.canvas.canvas_background import CanvasBackground
from dreamstack.raster.core.layer import (
    BlendMode,
    Layer,
    LayerBase,
    LayerGroup,
    apply_blend_mode,
)
from dreamstack.raster.core.pixel import BitDepth, PixelData, PixelFormat

if TYPE_CHECKING:
    from dreamstack.raster.core.image import Image


class Canvas:
    """
    Canvas for rendering and compositing layers.

    The Canvas handles:
    - Layer compositing with blend modes
    - Background rendering
    - Viewport management
    - Export to final image

    Example:
        >>> canvas = Canvas(1920, 1080)
        >>> canvas.layers.add(background_layer)
        >>> canvas.layers.add(foreground_layer)
        >>> result = canvas.render()
    """

    def __init__(
        self,
        width: int,
        height: int,
        pixel_format: PixelFormat = PixelFormat.RGBA,
        bit_depth: BitDepth = BitDepth.UINT8,
        dpi: tuple[float, float] = (72.0, 72.0),
        background: CanvasBackground | None = None,
    ):
        """
        Initialize canvas.

        Args:
            width: Canvas width in pixels
            height: Canvas height in pixels
            pixel_format: Default pixel format
            bit_depth: Default bit depth
            dpi: Resolution in dots per inch
            background: Background settings
        """
        self._width = width
        self._height = height
        self._pixel_format = pixel_format
        self._bit_depth = bit_depth
        self._dpi = dpi
        self._background = background or CanvasBackground()
        self._layers = LayerGroup(name="Root")
        self._cached_render: NDArray | None = None
        self._render_dirty = True

    @property
    def width(self) -> int:
        """Get canvas width."""
        return self._width

    @property
    def height(self) -> int:
        """Get canvas height."""
        return self._height

    @property
    def size(self) -> Size:
        """Get canvas size."""
        return Size(self._width, self._height)

    @property
    def bounds(self) -> Bounds:
        """Get canvas bounds."""
        return Bounds(0, 0, self._width, self._height)

    @property
    def pixel_format(self) -> PixelFormat:
        """Get default pixel format."""
        return self._pixel_format

    @property
    def bit_depth(self) -> BitDepth:
        """Get default bit depth."""
        return self._bit_depth

    @property
    def dpi(self) -> tuple[float, float]:
        """Get resolution."""
        return self._dpi

    @dpi.setter
    def dpi(self, value: tuple[float, float]) -> None:
        """Set resolution."""
        self._dpi = value

    @property
    def background(self) -> CanvasBackground:
        """Get background settings."""
        return self._background

    @background.setter
    def background(self, value: CanvasBackground) -> None:
        """Set background settings."""
        self._background = value
        self._render_dirty = True

    @property
    def layers(self) -> LayerGroup:
        """Get root layer group."""
        return self._layers

    @property
    def width_inches(self) -> float:
        """Get width in inches."""
        return self._width / self._dpi[0]

    @property
    def height_inches(self) -> float:
        """Get height in inches."""
        return self._height / self._dpi[1]

    @property
    def width_cm(self) -> float:
        """Get width in centimeters."""
        return self.width_inches * 2.54

    @property
    def height_cm(self) -> float:
        """Get height in centimeters."""
        return self.height_inches * 2.54

    def invalidate(self) -> None:
        """Mark canvas as needing re-render."""
        self._render_dirty = True
        self._cached_render = None

    def resize(self, width: int, height: int, anchor: str = "center") -> None:
        """
        Resize canvas (not image).

        Args:
            width: New width
            height: New height
            anchor: Anchor point (center, top-left, top-right, etc.)
        """
        old_w, old_h = self._width, self._height
        self._width = width
        self._height = height

        # Calculate offset for layers
        anchors = {
            "top-left": (0, 0),
            "top": ((width - old_w) // 2, 0),
            "top-right": (width - old_w, 0),
            "left": (0, (height - old_h) // 2),
            "center": ((width - old_w) // 2, (height - old_h) // 2),
            "right": (width - old_w, (height - old_h) // 2),
            "bottom-left": (0, height - old_h),
            "bottom": ((width - old_w) // 2, height - old_h),
            "bottom-right": (width - old_w, height - old_h),
        }

        offset = anchors.get(anchor, (0, 0))

        # Offset all layers
        for layer in self._layers.flatten_hierarchy():
            layer.offset = Point(layer.offset.x + offset[0], layer.offset.y + offset[1])

        self._render_dirty = True

    def crop(self, bounds: Bounds) -> None:
        """
        Crop canvas to bounds.

        Args:
            bounds: Crop region
        """
        self._width = int(bounds.width)
        self._height = int(bounds.height)

        # Offset layers
        for layer in self._layers.flatten_hierarchy():
            layer.offset = Point(layer.offset.x - bounds.x, layer.offset.y - bounds.y)

        self._render_dirty = True

    def render_background(self) -> NDArray:
        """
        Render canvas background.

        Returns:
            RGBA float array of background
        """
        result = np.zeros((self._height, self._width, 4), dtype=np.float32)

        if self._background.type == "transparent":
            # Already transparent
            pass

        elif self._background.type == "color":
            color = np.array(self._background.color1, dtype=np.float32) / 255
            result[:, :] = color

        elif self._background.type == "checker":
            # Create checker pattern
            color1 = np.array(self._background.color1, dtype=np.float32) / 255
            color2 = np.array(self._background.color2, dtype=np.float32) / 255
            size = self._background.checker_size

            for y in range(0, self._height, size):
                for x in range(0, self._width, size):
                    is_dark = ((x // size) + (y // size)) % 2 == 0
                    color = color2 if is_dark else color1
                    y2 = min(y + size, self._height)
                    x2 = min(x + size, self._width)
                    result[y:y2, x:x2] = color

        return result

    def render(self, include_background: bool = True) -> NDArray:
        """
        Render all layers to final image.

        Args:
            include_background: Whether to include background

        Returns:
            Composited RGBA float array
        """
        if not self._render_dirty and self._cached_render is not None:
            return self._cached_render.copy()

        canvas_size = self.size

        # Start with background
        if include_background:
            result = self.render_background()
        else:
            result = np.zeros((self._height, self._width, 4), dtype=np.float32)

        # Render and composite layers
        result = self._composite_layer(result, self._layers, canvas_size)

        self._cached_render = result
        self._render_dirty = False

        return result.copy()

    def _composite_layer(
        self, base: NDArray, layer: LayerBase, canvas_size: Size
    ) -> NDArray:
        """
        Composite a single layer onto base.

        Args:
            base: Base image to composite onto
            layer: Layer to composite
            canvas_size: Canvas size

        Returns:
            Composited result
        """
        if not layer.visible:
            return base

        if isinstance(layer, LayerGroup):
            if layer.blend_mode == BlendMode.PASS_THROUGH:
                # Composite children directly onto base
                for child in layer.children:
                    base = self._composite_layer(base, child, canvas_size)
                return base
            else:
                # Composite group first, then blend
                group_render = layer.render(canvas_size)
        else:
            group_render = layer.render(canvas_size)

        # Apply blend mode
        if layer.blend_mode == BlendMode.NORMAL:
            # Simple alpha compositing
            alpha = group_render[:, :, 3:4]
            result = base * (1 - alpha) + group_render * alpha
        else:
            # Apply blend mode to RGB
            blended = apply_blend_mode(
                base[:, :, :3], group_render[:, :, :3], layer.blend_mode
            )
            alpha = group_render[:, :, 3:4]
            result = base.copy()
            result[:, :, :3] = base[:, :, :3] * (1 - alpha) + blended * alpha
            result[:, :, 3:4] = base[:, :, 3:4] + alpha * (1 - base[:, :, 3:4])

        return result

    def flatten(self) -> Layer:
        """
        Flatten all layers into a single layer.

        Returns:
            Flattened Layer
        """
        rendered = self.render(include_background=True)

        # Convert to uint8
        uint8_data = (rendered * 255).clip(0, 255).astype(np.uint8)

        pixel_data = PixelData(
            data=uint8_data,
            pixel_format=PixelFormat.RGBA,
            bit_depth=BitDepth.UINT8,
        )

        return Layer(pixel_data, name="Flattened")

    def merge_visible(self) -> Layer:
        """
        Merge all visible layers into a single layer.

        Returns:
            Merged Layer
        """
        rendered = self.render(include_background=False)

        # Convert to uint8
        uint8_data = (rendered * 255).clip(0, 255).astype(np.uint8)

        pixel_data = PixelData(
            data=uint8_data,
            pixel_format=PixelFormat.RGBA,
            bit_depth=BitDepth.UINT8,
        )

        return Layer(pixel_data, name="Merged Visible")

    def to_image(self) -> Image:
        """
        Convert canvas to Image.

        Returns:
            Image with flattened content
        """
        from dreamstack.raster.core.image import Image, ImageMetadata

        rendered = self.render(include_background=True)

        # Convert to target bit depth
        if self._bit_depth == BitDepth.UINT8:
            data = (rendered * 255).clip(0, 255).astype(np.uint8)
        elif self._bit_depth == BitDepth.UINT16:
            data = (rendered * 65535).clip(0, 65535).astype(np.uint16)
        else:
            data = rendered

        pixel_data = PixelData(
            data=data,
            pixel_format=self._pixel_format,
            bit_depth=self._bit_depth,
        )

        metadata = ImageMetadata(dpi=self._dpi)

        return Image(pixel_data, metadata)

    def create_layer(
        self, name: str = "Layer", fill_color: tuple | None = None
    ) -> Layer:
        """
        Create a new layer matching canvas settings.

        Args:
            name: Layer name
            fill_color: Optional fill color

        Returns:
            New Layer
        """
        return Layer.create(
            width=self._width,
            height=self._height,
            pixel_format=self._pixel_format,
            bit_depth=self._bit_depth,
            fill_color=fill_color,
            name=name,
        )

    @classmethod
    def create(
        cls,
        width: int,
        height: int,
        pixel_format: PixelFormat = PixelFormat.RGBA,
        bit_depth: BitDepth = BitDepth.UINT8,
        dpi: tuple[float, float] = (72.0, 72.0),
        background_color: tuple | None = None,
    ) -> Canvas:
        """
        Create a new canvas.

        Args:
            width: Canvas width
            height: Canvas height
            pixel_format: Pixel format
            bit_depth: Bit depth
            dpi: Resolution
            background_color: Optional background color

        Returns:
            New Canvas
        """
        if background_color:
            background = CanvasBackground(type="color", color1=background_color)
        else:
            background = CanvasBackground(type="checker")

        return cls(
            width=width,
            height=height,
            pixel_format=pixel_format,
            bit_depth=bit_depth,
            dpi=dpi,
            background=background,
        )

    @classmethod
    def from_image(cls, image: Image) -> Canvas:
        """
        Create canvas from image.

        Args:
            image: Source image

        Returns:
            Canvas with image as background layer
        """
        canvas = cls(
            width=image.width,
            height=image.height,
            pixel_format=image.pixel_format,
            bit_depth=image.bit_depth,
            dpi=image.metadata.dpi,
        )

        # Add image as layer
        layer = Layer.from_image(image, name="Background")
        canvas.layers.add(layer)

        return canvas
