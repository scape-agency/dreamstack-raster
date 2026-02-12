"""
Merge Layers
============

Merge multiple layers with blend modes.

"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from dreamstack.raster.compositing.blend.blend_mode import BlendMode
from dreamstack.raster.compositing.blend.composite import composite


@dataclass
class LayerInfo:
    """Information about a layer for merging.

    Attributes:
        image: Layer image data.
        blend_mode: Blend mode for this layer.
        opacity: Layer opacity (0.0-1.0).
        position: Layer position (x, y).
        visible: Whether layer is visible.
        name: Optional layer name.
    """

    image: NDArray[np.uint8]
    blend_mode: BlendMode = BlendMode.NORMAL
    opacity: float = 1.0
    position: tuple[int, int] = (0, 0)
    visible: bool = True
    name: str | None = None


def merge_layers(
    layers: Sequence[LayerInfo],
    *,
    background_color: tuple[int, int, int, int] = (255, 255, 255, 255),
    output_size: tuple[int, int] | None = None,
) -> NDArray[np.uint8]:
    """Merge multiple layers into a single image.

    Layers are composited from bottom to top using their
    individual blend modes and opacities.

    Args:
        layers: Sequence of LayerInfo objects (bottom to top).
        background_color: Background fill color (BGRA).
        output_size: Output size (width, height). Auto-calculated if None.

    Returns:
        Merged image.

    Example:
        >>> layers = [
        ...     LayerInfo(bg_image),
        ...     LayerInfo(shadow, BlendMode.MULTIPLY, 0.5),
        ...     LayerInfo(highlight, BlendMode.SCREEN, 0.3),
        ... ]
        >>> result = merge_layers(layers)
    """
    if not layers:
        raise ValueError("At least one layer is required")

    # Calculate output size
    if output_size:
        w, h = output_size
    else:
        # Find bounds of all layers
        max_w, max_h = 0, 0
        for layer in layers:
            if layer.visible:
                lh, lw = layer.image.shape[:2]
                x, y = layer.position
                max_w = max(max_w, x + lw)
                max_h = max(max_h, y + lh)
        w, h = max(max_w, 1), max(max_h, 1)

    # Create background
    result = np.zeros((h, w, 4), dtype=np.uint8)
    result[:, :, 0] = background_color[0]
    result[:, :, 1] = background_color[1]
    result[:, :, 2] = background_color[2]
    result[:, :, 3] = background_color[3]

    # Composite each layer
    for layer in layers:
        if not layer.visible or layer.opacity <= 0:
            continue

        result = composite(
            result,
            layer.image,
            mode=layer.blend_mode,
            opacity=layer.opacity,
            position=layer.position,
        )

    return result
