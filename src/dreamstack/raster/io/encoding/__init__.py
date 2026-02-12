"""
Dreamstack Raster - IO Encoding Module
======================================

Base64 encoding utilities for web embedding.

"""

from __future__ import annotations

from dreamstack.raster.io.encoding.base64_to_image import base64_to_image
from dreamstack.raster.io.encoding.data_uri_to_image import data_uri_to_image
from dreamstack.raster.io.encoding.file_to_base64 import file_to_base64
from dreamstack.raster.io.encoding.file_to_data_uri import file_to_data_uri
from dreamstack.raster.io.encoding.image_to_base64 import image_to_base64
from dreamstack.raster.io.encoding.image_to_data_uri import image_to_data_uri

__all__: list[str] = [
    "image_to_base64",
    "image_to_data_uri",
    "file_to_base64",
    "file_to_data_uri",
    "base64_to_image",
    "data_uri_to_image",
]
