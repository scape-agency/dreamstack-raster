"""
Dreamstack Raster - Load PSD with Layers
========================================

Load PSD preserving layer structure.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dreamstack.raster.core.document import Document


def load_psd_with_layers(psd, name: str) -> Document:
    """Load PSD preserving layer structure."""
    from dreamstack.raster.core.bounds import Point
    from dreamstack.raster.core.canvas import Canvas
    from dreamstack.raster.core.document import Document
    from dreamstack.raster.core.layer import BlendMode, Layer, LayerGroup
    from dreamstack.raster.core.pixel import BitDepth, PixelData, PixelFormat

    # Create canvas
    canvas = Canvas(
        width=psd.width,
        height=psd.height,
        dpi=(psd.dpi, psd.dpi) if psd.dpi else (72.0, 72.0),
    )

    # PSD blend mode to our BlendMode mapping
    blend_mode_map = {
        "normal": BlendMode.NORMAL,
        "dissolve": BlendMode.DISSOLVE,
        "darken": BlendMode.DARKEN,
        "multiply": BlendMode.MULTIPLY,
        "color burn": BlendMode.COLOR_BURN,
        "linear burn": BlendMode.LINEAR_BURN,
        "lighten": BlendMode.LIGHTEN,
        "screen": BlendMode.SCREEN,
        "color dodge": BlendMode.COLOR_DODGE,
        "linear dodge": BlendMode.LINEAR_DODGE,
        "overlay": BlendMode.OVERLAY,
        "soft light": BlendMode.SOFT_LIGHT,
        "hard light": BlendMode.HARD_LIGHT,
        "vivid light": BlendMode.VIVID_LIGHT,
        "linear light": BlendMode.LINEAR_LIGHT,
        "pin light": BlendMode.PIN_LIGHT,
        "hard mix": BlendMode.HARD_MIX,
        "difference": BlendMode.DIFFERENCE,
        "exclusion": BlendMode.EXCLUSION,
        "subtract": BlendMode.SUBTRACT,
        "divide": BlendMode.DIVIDE,
        "hue": BlendMode.HUE,
        "saturation": BlendMode.SATURATION,
        "color": BlendMode.COLOR,
        "luminosity": BlendMode.LUMINOSITY,
    }

    def process_layer(psd_layer):
        """Process a PSD layer recursively."""
        if psd_layer.is_group():
            # Create layer group
            group = LayerGroup(
                name=psd_layer.name,
                opacity=psd_layer.opacity / 255.0,
                blend_mode=blend_mode_map.get(
                    psd_layer.blend_mode.lower(), BlendMode.NORMAL
                ),
                visible=psd_layer.visible,
            )

            for child in psd_layer:
                child_layer = process_layer(child)
                if child_layer:
                    group.add(child_layer)

            return group
        else:
            # Regular layer
            pil_image = psd_layer.composite()
            if pil_image is None:
                return None

            array = np.array(pil_image)

            if pil_image.mode == "RGBA":
                pixel_format = PixelFormat.RGBA
            else:
                array = np.array(pil_image.convert("RGBA"))
                pixel_format = PixelFormat.RGBA

            if array.ndim == 2:
                array = array[:, :, np.newaxis]

            pixel_data = PixelData(
                data=array, pixel_format=pixel_format, bit_depth=BitDepth.UINT8
            )

            layer = Layer(
                pixel_data=pixel_data,
                name=psd_layer.name,
                opacity=psd_layer.opacity / 255.0,
                blend_mode=blend_mode_map.get(
                    psd_layer.blend_mode.lower(), BlendMode.NORMAL
                ),
                visible=psd_layer.visible,
            )

            # Set layer offset
            layer.offset = Point(psd_layer.left, psd_layer.top)

            return layer

    # Process all layers
    for psd_layer in psd:
        layer = process_layer(psd_layer)
        if layer:
            canvas.layers.add(layer)

    return Document(canvas, name=name)
