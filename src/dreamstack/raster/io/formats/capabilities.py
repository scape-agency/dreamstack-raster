# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Format Capabilities
=======================================

Dataclass and function for format capabilities.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from dataclasses import dataclass

from dreamstack.raster.io.formats.constants import (
    ALPHA_FORMATS,
    HIGH_BIT_DEPTH_FORMATS,
    LAYER_FORMATS,
    READ_FORMATS,
    WRITE_FORMATS,
)
from dreamstack.raster.io.formats.image_format import ImageFormat


@dataclass
class FormatCapabilities:
    """
    Describes capabilities of an image format.

    Attributes:
        format: The image format
        can_read: Whether reading is supported
        can_write: Whether writing is supported
        supports_alpha: Whether alpha channel is supported
        supports_layers: Whether multiple layers are supported
        supports_16bit: Whether 16-bit depth is supported
        supports_float: Whether floating point is supported
        supports_animation: Whether animation is supported
        supports_metadata: Whether metadata is supported
        lossy: Whether format uses lossy compression
    """

    format: ImageFormat
    can_read: bool = True
    can_write: bool = True
    supports_alpha: bool = True
    supports_layers: bool = False
    supports_16bit: bool = False
    supports_float: bool = False
    supports_animation: bool = False
    supports_metadata: bool = True
    lossy: bool = False


def get_capabilities(image_format: ImageFormat) -> FormatCapabilities:
    """
    Get capabilities for a format.

    Args:
        image_format: Image format

    Returns:
        Format capabilities
    """
    caps = FormatCapabilities(
        format=image_format,
        can_read=image_format in READ_FORMATS,
        can_write=image_format in WRITE_FORMATS,
        supports_alpha=image_format in ALPHA_FORMATS,
        supports_layers=image_format in LAYER_FORMATS,
        supports_16bit=image_format in HIGH_BIT_DEPTH_FORMATS,
    )

    # Format-specific settings
    if image_format == ImageFormat.JPEG:
        caps.supports_alpha = False
        caps.lossy = True
    elif image_format == ImageFormat.GIF:
        caps.supports_animation = True
    elif image_format == ImageFormat.WEBP:
        caps.supports_animation = True
        caps.lossy = True
    elif image_format == ImageFormat.EXR:
        caps.supports_float = True
    elif image_format == ImageFormat.HDR:
        caps.supports_float = True
        caps.supports_alpha = False
    elif image_format in (ImageFormat.PSD, ImageFormat.PSB):
        caps.supports_float = False

    return caps
