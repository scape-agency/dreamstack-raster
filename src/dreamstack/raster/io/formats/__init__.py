"""
Dreamstack Raster - Formats Module
==================================

Image format definitions and constants.

"""

from dreamstack.raster.io.formats.capabilities import (
    FormatCapabilities,
    get_capabilities,
)
from dreamstack.raster.io.formats.constants import (
    ALPHA_FORMATS,
    FORMAT_EXTENSIONS,
    FORMAT_MIME_TYPES,
    HIGH_BIT_DEPTH_FORMATS,
    LAYER_FORMATS,
    READ_FORMATS,
    WRITE_FORMATS,
)
from dreamstack.raster.io.formats.detection import (
    get_format_for_mime,
    get_format_for_path,
)
from dreamstack.raster.io.formats.extensions import (
    get_extensions,
    get_primary_extension,
)
from dreamstack.raster.io.formats.image_format import ImageFormat
from dreamstack.raster.io.formats.queries import (
    get_read_formats,
    get_supported_formats,
    get_write_formats,
    supports_alpha,
    supports_high_bit_depth,
    supports_layers,
)

__all__ = [
    "ImageFormat",
    "FORMAT_EXTENSIONS",
    "FORMAT_MIME_TYPES",
    "READ_FORMATS",
    "WRITE_FORMATS",
    "LAYER_FORMATS",
    "ALPHA_FORMATS",
    "HIGH_BIT_DEPTH_FORMATS",
    "get_format_for_path",
    "get_format_for_mime",
    "get_extensions",
    "get_primary_extension",
    "FormatCapabilities",
    "get_capabilities",
    "get_supported_formats",
    "get_read_formats",
    "get_write_formats",
    "supports_layers",
    "supports_alpha",
    "supports_high_bit_depth",
]
