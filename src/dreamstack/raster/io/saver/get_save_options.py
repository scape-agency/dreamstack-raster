# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Get Save Options
====================================

Get available save options for a format.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from typing import Any

from dreamstack.raster.io.formats import ImageFormat


def get_save_options(image_format: ImageFormat) -> dict[str, Any]:
    """
    Get available save options for a format.

    Args:
        image_format: Image format

    Returns:
        Dictionary describing available options
    """
    options = {}

    if image_format == ImageFormat.PNG:
        options = {
            "compression": {
                "type": "int",
                "range": (0, 9),
                "default": 6,
                "description": "Compression level",
            },
            "optimize": {
                "type": "bool",
                "default": False,
                "description": "Optimize for size",
            },
        }

    elif image_format == ImageFormat.JPEG:
        options = {
            "quality": {
                "type": "int",
                "range": (1, 100),
                "default": 95,
                "description": "JPEG quality",
            },
            "progressive": {
                "type": "bool",
                "default": False,
                "description": "Progressive JPEG",
            },
            "optimize": {
                "type": "bool",
                "default": True,
                "description": "Optimize Huffman tables",
            },
        }

    elif image_format == ImageFormat.WEBP:
        options = {
            "quality": {
                "type": "int",
                "range": (1, 100),
                "default": 80,
                "description": "WebP quality",
            },
            "lossless": {
                "type": "bool",
                "default": False,
                "description": "Use lossless compression",
            },
        }

    elif image_format == ImageFormat.TIFF:
        options = {
            "compression": {
                "type": "enum",
                "values": ["none", "lzw", "zip", "jpeg"],
                "default": "lzw",
                "description": "Compression method",
            }
        }

    elif image_format == ImageFormat.EXR:
        options = {
            "compression": {
                "type": "enum",
                "values": ["none", "zip", "piz", "pxr24", "b44", "b44a"],
                "default": "zip",
                "description": "Compression method",
            }
        }

    return options
