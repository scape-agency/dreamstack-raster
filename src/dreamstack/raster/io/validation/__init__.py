"""
File and Image Validation Utilities
===================================

Validation functions for paths, files, and image data.

"""

from __future__ import annotations

from dreamstack.raster.io.validation.constants import (
    SUPPORTED_EXTENSIONS,
    SUPPORTED_RAW_EXTENSIONS,
)
from dreamstack.raster.io.validation.ensure_directory import ensure_directory
from dreamstack.raster.io.validation.get_image_files import get_image_files
from dreamstack.raster.io.validation.is_valid_image_file import (
    is_valid_image_file,
)
from dreamstack.raster.io.validation.validate_directory import (
    validate_directory,
)
from dreamstack.raster.io.validation.validate_file import validate_file
from dreamstack.raster.io.validation.validate_image_array import (
    validate_image_array,
)
from dreamstack.raster.io.validation.validate_image_extension import (
    validate_image_extension,
)
from dreamstack.raster.io.validation.validate_image_file import (
    validate_image_file,
)
from dreamstack.raster.io.validation.validate_path import validate_path

__all__: list[str] = [
    # Constants
    "SUPPORTED_EXTENSIONS",
    "SUPPORTED_RAW_EXTENSIONS",
    # Validation functions
    "validate_path",
    "validate_file",
    "validate_directory",
    "validate_image_extension",
    "validate_image_file",
    "validate_image_array",
    "is_valid_image_file",
    "get_image_files",
    "ensure_directory",
]
