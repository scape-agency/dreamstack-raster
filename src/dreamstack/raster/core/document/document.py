# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Document
============================

Document class managing the complete editing context.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from dreamstack.raster.core.bounds import Bounds, Point, Size
from dreamstack.raster.core.canvas import Canvas
from dreamstack.raster.core.document.grid_settings import GridSettings
from dreamstack.raster.core.document.guide import Guide
from dreamstack.raster.core.history import History
from dreamstack.raster.core.layer import (
    BlendMode,
    Layer,
    LayerBase,
    LayerGroup,
)
from dreamstack.raster.core.pixel import (
    CHANNEL_COUNT,
    DTYPE_MAP,
    BitDepth,
    PixelData,
    PixelFormat,
)
from dreamstack.raster.io import load_image, save_image

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.image import Image

    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.selection import Selection


class Document:
    """
    Document class representing a complete editing session.

    The Document manages:
    - Canvas and layers
    - History/undo system
    - Selection state
    - Guides and grids
    - File save/load

    Example:
        >>> doc = Document.create(1920, 1080, name="My Project")
        >>> doc.add_layer(image_layer)
        >>> doc.history.push_state("Add Layer", doc.serialize())
        >>> doc.save("project.drst")
    """

    def __init__(
        self,
        canvas: Canvas,
        name: str = "Untitled",
        path: Path | None = None,
    ):
        """
        Initialize document.

        Args:
            canvas: Canvas for the document
            name: Document name
            path: Optional file path
        """
        self._canvas = canvas
        self._name = name
        self._path = path
        self._history = History(max_states=100)
        self._selection: Selection | None = None
        self._guides: list[Guide] = []
        self._grids: GridSettings = GridSettings()
        self._metadata: dict[str, Any] = {}
        self._active_layer: LayerBase | None = None
        self._color_profile: str = "sRGB"

    @property
    def canvas(self) -> Canvas:
        """Get document canvas."""
        return self._canvas

    @property
    def name(self) -> str:
        """Get document name."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """Set document name."""
        self._name = value

    @property
    def path(self) -> Path | None:
        """Get file path."""
        return self._path

    @property
    def history(self) -> History:
        """Get history manager."""
        return self._history

    @property
    def layers(self) -> LayerGroup:
        """Get root layer group."""
        return self._canvas.layers

    @property
    def selection(self) -> Selection | None:
        """Get current selection."""
        return self._selection

    @selection.setter
    def selection(self, value: Selection | None) -> None:
        """Set current selection."""
        self._selection = value

    @property
    def active_layer(self) -> LayerBase | None:
        """Get active layer."""
        return self._active_layer

    @active_layer.setter
    def active_layer(self, value: LayerBase | None) -> None:
        """Set active layer."""
        self._active_layer = value

    @property
    def width(self) -> int:
        """Get document width."""
        return self._canvas.width

    @property
    def height(self) -> int:
        """Get document height."""
        return self._canvas.height

    @property
    def size(self) -> Size:
        """Get document size."""
        return self._canvas.size

    @property
    def is_modified(self) -> bool:
        """Check if document has unsaved changes."""
        return self._history.is_modified

    @property
    def guides(self) -> list[Guide]:
        """Get guides."""
        return self._guides.copy()

    @property
    def grid_settings(self) -> GridSettings:
        """Get grid settings."""
        return self._grids

    @property
    def color_profile(self) -> str:
        """Get color profile."""
        return self._color_profile

    @color_profile.setter
    def color_profile(self, value: str) -> None:
        """Set color profile."""
        self._color_profile = value

    # =========================================================================
    # Layer Management
    # =========================================================================

    def add_layer(
        self,
        layer: LayerBase,
        index: int | None = None,
        parent: LayerGroup | None = None,
    ) -> None:
        """
        Add a layer to the document.

        Args:
            layer: Layer to add
            index: Position in layer stack
            parent: Parent group (default: root)
        """
        target = parent or self._canvas.layers
        target.add(layer, index)

        if self._active_layer is None:
            self._active_layer = layer

        self._canvas.invalidate()

    def remove_layer(self, layer: LayerBase | str) -> LayerBase:
        """
        Remove a layer from the document.

        Args:
            layer: Layer or layer ID to remove

        Returns:
            Removed layer
        """
        layer_obj: LayerBase
        if isinstance(layer, str):
            found = self.find_layer(layer)
            if found is None:
                raise ValueError(f"Layer not found: {layer}")
            layer_obj = found
        else:
            layer_obj = layer

        parent = layer_obj.parent or self._canvas.layers
        removed = parent.remove(layer_obj)

        if self._active_layer == removed:
            # Select next available layer
            all_layers = self._canvas.layers.flatten_hierarchy()
            self._active_layer = all_layers[0] if all_layers else None

        self._canvas.invalidate()
        return removed

    def duplicate_layer(self, layer: LayerBase | str) -> LayerBase:
        """
        Duplicate a layer.

        Args:
            layer: Layer or ID to duplicate

        Returns:
            New duplicated layer
        """
        resolved_layer: LayerBase
        if isinstance(layer, str):
            found = self.find_layer(layer)
            if found is None:
                raise ValueError(f"Layer not found: {layer}")
            resolved_layer = found
        else:
            resolved_layer = layer

        copy = resolved_layer.copy()
        parent = resolved_layer.parent or self._canvas.layers

        # Insert above original
        index = parent.children.index(resolved_layer)
        parent.add(copy, index + 1)

        self._active_layer = copy
        self._canvas.invalidate()

        return copy

    def merge_layers(self, layers: list[LayerBase]) -> Layer:
        """
        Merge multiple layers into one.

        Args:
            layers: Layers to merge

        Returns:
            Merged layer
        """
        if not layers:
            raise ValueError("No layers to merge")

        # Render layers to single image
        result = np.zeros((self.height, self.width, 4), dtype=np.float32)

        for layer in layers:
            layer_render = layer.render(self.size)
            alpha = layer_render[:, :, 3:4]
            result = result * (1 - alpha) + layer_render * alpha

        # Convert to uint8
        uint8_data = (result * 255).clip(0, 255).astype(np.uint8)
        pixel_data = PixelData(
            data=uint8_data,
            pixel_format=PixelFormat.RGBA,
            bit_depth=BitDepth.UINT8,
        )

        merged = Layer(pixel_data, name="Merged")

        # Find position of first layer
        first_layer = layers[0]
        parent = first_layer.parent or self._canvas.layers
        index = parent.children.index(first_layer)

        # Remove old layers
        for layer in layers:
            self.remove_layer(layer)

        # Add merged layer
        parent.add(merged, index)
        self._active_layer = merged
        self._canvas.invalidate()

        return merged

    def flatten(self) -> None:
        """Flatten all layers into one."""
        flat = self._canvas.flatten()
        self._canvas.layers._children.clear()  # pylint: disable=protected-access
        self._canvas.layers.add(flat)
        self._active_layer = flat
        self._canvas.invalidate()

    def find_layer(self, identifier: str) -> LayerBase | None:
        """
        Find layer by name or ID.

        Args:
            identifier: Layer name or ID

        Returns:
            Found layer or None
        """
        # Try ID first
        found = self._canvas.layers.find_by_id(identifier)
        if found:
            return found

        # Try name
        return self._canvas.layers.find(identifier)

    # =========================================================================
    # Guide Management
    # =========================================================================

    def add_guide(self, guide: Guide) -> None:
        """Add a guide."""
        self._guides.append(guide)

    def remove_guide(self, guide: Guide) -> None:
        """Remove a guide."""
        self._guides.remove(guide)

    def clear_guides(self) -> None:
        """Remove all guides."""
        self._guides.clear()

    # =========================================================================
    # Document Operations
    # =========================================================================

    def resize_canvas(
        self, width: int, height: int, anchor: str = "center"
    ) -> None:
        """
        Resize document canvas.

        Args:
            width: New width
            height: New height
            anchor: Anchor point for resize
        """
        self._canvas.resize(width, height, anchor)

    def resize_image(
        self, width: int, height: int, method: str = "lanczos"
    ) -> None:
        """
        Resize entire document including layers.

        Args:
            width: New width
            height: New height
            method: Interpolation method
        """
        import cv2  # pylint: disable=import-outside-toplevel

        scale_x = width / self._canvas.width
        scale_y = height / self._canvas.height

        # Resize each layer
        for layer in self._canvas.layers.flatten_hierarchy():
            if isinstance(layer, Layer):
                # Resize pixel data
                # pylint: disable=no-member
                interpolation = {
                    "nearest": cv2.INTER_NEAREST,
                    "bilinear": cv2.INTER_LINEAR,
                    "bicubic": cv2.INTER_CUBIC,
                    "lanczos": cv2.INTER_LANCZOS4,
                }.get(method, cv2.INTER_LANCZOS4)
                # pylint: enable=no-member

                new_w = int(layer.width * scale_x)
                new_h = int(layer.height * scale_y)

                resized = cv2.resize(  # pylint: disable=no-member  # pylint: disable=no-member
                    layer.pixel_data.data,
                    (new_w, new_h),
                    interpolation=interpolation,
                )

                if resized.ndim == 2:
                    resized = resized[:, :, np.newaxis]

                layer._pixel_data = (  # pylint: disable=protected-access
                    PixelData(
                        data=resized,
                        pixel_format=layer.pixel_data.pixel_format,
                        bit_depth=layer.pixel_data.bit_depth,
                    )
                )

            # Scale offset
            layer.offset = Point(
                layer.offset.x * scale_x, layer.offset.y * scale_y
            )

        # Update canvas size
        # pylint: disable=protected-access
        self._canvas._width = width
        self._canvas._height = height
        # pylint: enable=protected-access
        self._canvas.invalidate()

    def crop(self, bounds: Bounds) -> None:
        """
        Crop document to bounds.

        Args:
            bounds: Crop region
        """
        self._canvas.crop(bounds)

    def render(self) -> Image:
        """
        Render document to image.

        Returns:
            Rendered Image
        """
        return self._canvas.to_image()

    # =========================================================================
    # Serialization
    # =========================================================================

    def serialize(self) -> bytes:
        """
        Serialize document to bytes.

        Returns:
            Serialized document data
        """
        # Simple pickle-based serialization
        # In production, use a proper format
        data = {
            "name": self._name,
            "width": self._canvas.width,
            "height": self._canvas.height,
            "pixel_format": self._canvas.pixel_format,
            "bit_depth": self._canvas.bit_depth,
            "dpi": self._canvas.dpi,
            "layers": self._serialize_layer_group(self._canvas.layers),
            "guides": [
                (g.position, g.orientation, g.color) for g in self._guides
            ],
            "grids": {
                "enabled": self._grids.enabled,
                "size": self._grids.size,
                "subdivisions": self._grids.subdivisions,
            },
            "metadata": self._metadata,
            "color_profile": self._color_profile,
        }

        return pickle.dumps(data)

    def _serialize_layer_group(self, group: LayerGroup) -> dict:
        """Serialize a layer group."""
        return {
            "type": "group",
            "name": group.name,
            "opacity": group.opacity,
            "blend_mode": group.blend_mode.name,
            "visible": group.visible,
            "locked": group.locked,
            "children": [
                self._serialize_layer(child) for child in group.children
            ],
        }

    def _serialize_layer(self, layer: LayerBase) -> dict:
        """Serialize a layer."""
        if isinstance(layer, LayerGroup):
            return self._serialize_layer_group(layer)
        elif isinstance(layer, Layer):
            return {
                "type": "layer",
                "name": layer.name,
                "opacity": layer.opacity,
                "blend_mode": layer.blend_mode.name,
                "visible": layer.visible,
                "locked": layer.locked,
                "offset": (layer.offset.x, layer.offset.y),
                "pixel_data": layer.pixel_data.data.tobytes(),
                "pixel_format": layer.pixel_data.pixel_format.name,
                "bit_depth": layer.pixel_data.bit_depth.name,
                "width": layer.width,
                "height": layer.height,
            }
        else:
            return {
                "type": "unknown",
                "name": layer.name,
            }

    def deserialize(self, data: bytes) -> None:
        """
        Deserialize document from bytes.

        Args:
            data: Serialized document data
        """
        doc_data = pickle.loads(data)

        self._name = doc_data["name"]
        # pylint: disable=protected-access
        self._canvas._width = doc_data["width"]
        self._canvas._height = doc_data["height"]
        self._canvas._pixel_format = doc_data["pixel_format"]
        self._canvas._bit_depth = doc_data["bit_depth"]
        self._canvas._dpi = doc_data["dpi"]

        # Deserialize layers
        self._canvas._layers._children.clear()
        for child_data in doc_data["layers"]["children"]:
            layer = self._deserialize_layer(child_data)
            if layer:
                self._canvas._layers.add(layer)
        # pylint: enable=protected-access

        # Deserialize guides
        self._guides.clear()
        for pos, orientation, color in doc_data.get("guides", []):
            self._guides.append(Guide(pos, orientation, color))

        # Deserialize grids
        grid_data = doc_data.get("grids", {})
        self._grids.enabled = grid_data.get("enabled", False)
        self._grids.size = grid_data.get("size", 50)
        self._grids.subdivisions = grid_data.get("subdivisions", 1)

        self._metadata = doc_data.get("metadata", {})
        self._color_profile = doc_data.get("color_profile", "sRGB")

        self._canvas.invalidate()

    def _deserialize_layer(self, data: dict) -> LayerBase | None:
        """Deserialize a layer from dict."""
        layer_type = data.get("type")

        if layer_type == "group":
            group = LayerGroup(
                name=data["name"],
                opacity=data["opacity"],
                blend_mode=BlendMode[data["blend_mode"]],
                visible=data["visible"],
                locked=data["locked"],
            )
            for child_data in data.get("children", []):
                child = self._deserialize_layer(child_data)
                if child:
                    group.add(child)
            return group

        elif layer_type == "layer":
            pixel_format = PixelFormat[data["pixel_format"]]
            bit_depth = BitDepth[data["bit_depth"]]

            dtype = DTYPE_MAP[bit_depth]
            channels = CHANNEL_COUNT[pixel_format]

            pixel_array = np.frombuffer(data["pixel_data"], dtype=dtype)
            pixel_array = pixel_array.reshape(
                (data["height"], data["width"], channels)
            )

            pixel_data = PixelData(
                data=pixel_array.copy(),
                pixel_format=pixel_format,
                bit_depth=bit_depth,
            )

            layer = Layer(
                pixel_data=pixel_data,
                name=data["name"],
                opacity=data["opacity"],
                blend_mode=BlendMode[data["blend_mode"]],
                visible=data["visible"],
                locked=data["locked"],
            )
            layer.offset = Point(data["offset"][0], data["offset"][1])

            return layer

        return None

    # =========================================================================
    # File Operations
    # =========================================================================

    def save(self, path: str | Path | None = None) -> None:
        """
        Save document to file.

        Args:
            path: Output path (uses existing path if None)
        """
        if path is None:
            if self._path is None:
                raise ValueError("No path specified")
            path = self._path

        path = Path(path)
        self._path = path

        # Native format
        if path.suffix.lower() == ".drst":
            with open(path, "wb") as f:
                f.write(self.serialize())
        else:
            # Export as image
            image = self.render()
            save_image(image, path)

        self._history.mark_saved()

    @classmethod
    def load(cls, path: str | Path) -> Document:
        """
        Load document from file.

        Args:
            path: Input path

        Returns:
            Loaded Document
        """
        path = Path(path)

        if path.suffix.lower() == ".drst":
            # Native format
            with open(path, "rb") as f:
                data = f.read()

            # Create empty document and deserialize
            canvas = Canvas(1, 1)
            doc = cls(canvas, name=path.stem, path=path)
            doc.deserialize(data)
            return doc
        else:
            # Load as image
            image = load_image(path)

            canvas = Canvas.from_image(image)
            return cls(canvas, name=path.stem, path=path)

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @classmethod
    def create(
        cls,
        width: int,
        height: int,
        pixel_format: PixelFormat = PixelFormat.RGBA,
        bit_depth: BitDepth = BitDepth.UINT8,
        dpi: tuple = (72.0, 72.0),
        name: str = "Untitled",
        background_color: tuple | None = None,
    ) -> Document:
        """
        Create a new document.

        Args:
            width: Document width
            height: Document height
            pixel_format: Pixel format
            bit_depth: Bit depth
            dpi: Resolution
            name: Document name
            background_color: Optional background color

        Returns:
            New Document
        """
        canvas = Canvas.create(
            width=width,
            height=height,
            pixel_format=pixel_format,
            bit_depth=bit_depth,
            dpi=dpi,
            background_color=background_color,
        )

        doc = cls(canvas, name=name)

        # Create initial layer
        if background_color:
            layer = canvas.create_layer("Background", background_color)
        else:
            layer = canvas.create_layer("Layer 1")

        canvas.layers.add(layer)
        doc._active_layer = layer

        # Save initial state
        doc._history.push_state("New Document", doc.serialize())
        doc._history.mark_saved()

        return doc

    @classmethod
    def from_image(cls, image: Image, name: str | None = None) -> Document:
        """
        Create document from image.

        Args:
            image: Source image
            name: Optional document name

        Returns:
            New Document
        """
        canvas = Canvas.from_image(image)
        doc = cls(canvas, name=name or image.name)

        if canvas.layers.children:
            doc._active_layer = canvas.layers.children[0]

        doc._history.push_state("Open Image", doc.serialize())
        doc._history.mark_saved()

        return doc
