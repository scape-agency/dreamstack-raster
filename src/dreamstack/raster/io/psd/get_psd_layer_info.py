# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Get PSD Layer Info
======================================

Get information about layers in a PSD file.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from pathlib import Path


def get_psd_layer_info(path: str | Path) -> list[dict]:
    """
    Get information about layers in a PSD file.

    Args:
        path: Path to PSD file

    Returns:
        List of layer information dictionaries
    """
    from psd_tools import PSDImage

    psd = PSDImage.open(path)

    def process_layer(layer, depth=0):
        info = {
            "name": layer.name,
            "visible": layer.visible,
            "opacity": layer.opacity / 255.0,
            "blend_mode": layer.blend_mode,
            "is_group": layer.is_group(),
            "bounds": (layer.left, layer.top, layer.right, layer.bottom),
            "depth": depth,
        }

        if layer.is_group():
            info["children"] = [
                process_layer(child, depth + 1) for child in layer
            ]

        return info

    return [process_layer(layer) for layer in psd]
