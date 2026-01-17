# -*- coding: utf-8 -*-

"""
Dreamstack Raster - Layer Group
===============================

Layer group containing multiple layers.

"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from dreamstack.raster.core.bounds import Bounds, Size
from dreamstack.raster.core.layer.blend import apply_blend_mode
from dreamstack.raster.core.layer.blend_mode import BlendMode
from dreamstack.raster.core.layer.layer_base import LayerBase
from dreamstack.raster.core.pixel import BitDepth, PixelData, PixelFormat


class LayerGroup(LayerBase):
    """
    Layer group containing multiple layers.

    Groups allow organizing layers and applying effects to multiple
    layers at once. Pass-through mode makes the group transparent
    for blending, while other modes composite the group first.
    """

    def __init__(
        self,
        name: str = "Group",
        opacity: float = 1.0,
        blend_mode: BlendMode = BlendMode.PASS_THROUGH,
        visible: bool = True,
        locked: bool = False,
    ):
        """
        Initialize a layer group.

        Args:
            name: Group name
            opacity: Group opacity
            blend_mode: Blend mode (PASS_THROUGH for transparent grouping)
            visible: Visibility
            locked: Lock state
        """
        super().__init__(name, opacity, blend_mode, visible, locked)
        self._children: List[LayerBase] = []
        self._expanded = True

    @property
    def children(self) -> List[LayerBase]:
        """Get child layers."""
        return self._children.copy()

    @property
    def expanded(self) -> bool:
        """Check if group is expanded in UI."""
        return self._expanded

    @expanded.setter
    def expanded(self, value: bool) -> None:
        """Set expanded state."""
        self._expanded = value

    @property
    def bounds(self) -> Bounds:
        """Get combined bounds of all children."""
        if not self._children:
            return Bounds(0, 0, 0, 0)

        result = self._children[0].bounds
        for child in self._children[1:]:
            result = result.union(child.bounds)

        return result

    def __len__(self) -> int:
        return len(self._children)

    def __getitem__(self, index: int) -> LayerBase:
        return self._children[index]

    def __iter__(self):
        return iter(self._children)

    def add(self, layer: LayerBase, index: int | None = None) -> None:
        """
        Add a layer to the group.

        Args:
            layer: Layer to add
            index: Optional position (default: top)
        """
        if layer._parent is not None:
            layer._parent.remove(layer)

        layer._parent = self

        if index is None:
            self._children.append(layer)
        else:
            self._children.insert(index, layer)

    def remove(self, layer: LayerBase | int) -> LayerBase:
        """
        Remove a layer from the group.

        Args:
            layer: Layer or index to remove

        Returns:
            Removed layer
        """
        if isinstance(layer, int):
            removed = self._children.pop(layer)
        else:
            self._children.remove(layer)
            removed = layer

        removed._parent = None
        return removed

    def move(self, layer: LayerBase | int, new_index: int) -> None:
        """
        Move a layer within the group.

        Args:
            layer: Layer or index to move
            new_index: New position
        """
        if isinstance(layer, int):
            layer = self._children[layer]

        old_index = self._children.index(layer)
        self._children.pop(old_index)

        # Adjust new index if needed
        if new_index > old_index:
            new_index -= 1

        self._children.insert(new_index, layer)

    def find(self, name: str) -> LayerBase | None:
        """
        Find layer by name.

        Args:
            name: Layer name

        Returns:
            Found layer or None
        """
        for child in self._children:
            if child.name == name:
                return child
            if isinstance(child, LayerGroup):
                found = child.find(name)
                if found:
                    return found
        return None

    def find_by_id(self, layer_id: str) -> LayerBase | None:
        """
        Find layer by ID.

        Args:
            layer_id: Layer ID

        Returns:
            Found layer or None
        """
        for child in self._children:
            if child.id == layer_id:
                return child
            if isinstance(child, LayerGroup):
                found = child.find_by_id(layer_id)
                if found:
                    return found
        return None

    def flatten_hierarchy(self) -> List[LayerBase]:
        """
        Get flattened list of all layers.

        Returns:
            List of all layers including nested ones
        """
        result = []
        for child in self._children:
            result.append(child)
            if isinstance(child, LayerGroup):
                result.extend(child.flatten_hierarchy())
        return result

    def render(self, canvas_size: Size) -> NDArray:
        """
        Render group to canvas size.

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

        # Composite children from bottom to top
        result = np.zeros(
            (int(canvas_size.height), int(canvas_size.width), 4),
            dtype=np.float32,
        )

        for child in self._children:
            if not child.visible:
                continue

            child_render = child.render(canvas_size)

            # Apply blend mode
            if (
                child.blend_mode == BlendMode.NORMAL
                or child.blend_mode == BlendMode.PASS_THROUGH
            ):
                # Simple alpha compositing
                child_alpha = child_render[:, :, 3:4]
                result = (
                    result * (1 - child_alpha) + child_render * child_alpha
                )
            else:
                # Apply blend mode to RGB, then composite
                blended = apply_blend_mode(
                    result[:, :, :3], child_render[:, :, :3], child.blend_mode
                )
                child_alpha = child_render[:, :, 3:4]
                result[:, :, :3] = (
                    result[:, :, :3] * (1 - child_alpha)
                    + blended * child_alpha
                )
                result[:, :, 3:4] = result[:, :, 3:4] + child_alpha * (
                    1 - result[:, :, 3:4]
                )

        # Apply mask
        result = self.apply_mask(result)

        # Apply opacity
        result[:, :, 3] *= self._opacity

        return result

    def copy(self) -> LayerGroup:
        """Create a copy of this group and its children."""
        new_group = LayerGroup(
            name=f"{self._name} copy",
            opacity=self._opacity,
            blend_mode=self._blend_mode,
            visible=self._visible,
            locked=self._locked,
        )

        for child in self._children:
            new_group.add(child.copy())

        if self._mask is not None:
            new_group._mask = self._mask.copy()

        return new_group

    def merge(self, canvas_size: Size):
        """
        Merge group into a single layer.

        Args:
            canvas_size: Canvas size for rendering

        Returns:
            Merged Layer
        """
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
