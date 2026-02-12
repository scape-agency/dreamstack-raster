# -*- coding: utf-8 -*-

"""
Dreamstack Raster - IO Module
=============================

File loading and saving for various image formats.

"""

from __future__ import annotations

from dreamstack.raster.io.compress import (
    CompressionConfig,
    CompressionResult,
    compress_image,
    compress_to_size,
    estimate_file_size,
    optimize_for_web,
)
from dreamstack.raster.io.encoding import (
    base64_to_image,
    data_uri_to_image,
    file_to_base64,
    file_to_data_uri,
    image_to_base64,
    image_to_data_uri,
)
from dreamstack.raster.io.exr import load_exr, save_exr
from dreamstack.raster.io.formats import (
    ImageFormat,
    get_format_for_path,
    get_read_formats,
    get_supported_formats,
    get_write_formats,
)
from dreamstack.raster.io.loader import load_image
from dreamstack.raster.io.metadata import read_metadata, write_metadata
from dreamstack.raster.io.psd import load_psd, save_psd
from dreamstack.raster.io.raw import load_raw
from dreamstack.raster.io.saver import save_image

__all__: list[str] = [
    "load_image",
    "save_image",
    "ImageFormat",
    "get_format_for_path",
    "get_supported_formats",
    "get_read_formats",
    "get_write_formats",
    "load_psd",
    "save_psd",
    "load_raw",
    "load_exr",
    "save_exr",
    "read_metadata",
    "write_metadata",
    # Compression
    "compress_image",
    "compress_to_size",
    "estimate_file_size",
    "optimize_for_web",
    "CompressionConfig",
    "CompressionResult",
    # Encoding
    "image_to_base64",
    "image_to_data_uri",
    "file_to_base64",
    "file_to_data_uri",
    "base64_to_image",
    "data_uri_to_image",
]
