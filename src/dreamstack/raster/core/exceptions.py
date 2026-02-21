# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Custom Exceptions
=====================================

Custom exception classes for error handling throughout the library.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

__all__: list[str] = [
    # Base Exceptions
    "DreamstackRasterError",
    # File/Path Exceptions
    "PathNotFoundError",
    "NotAFileError",
    "DirectoryNotFoundError",
    "InvalidPathError",
    # Image Exceptions
    "InvalidImageError",
    "InvalidImageTypeError",
    "UnsupportedFormatError",
    "ImageProcessingError",
    "ImageReadError",
    "ImageWriteError",
    # Parameter Exceptions
    "InvalidParameterError",
    "MutualExclusionError",
    "MissingRequiredParameterError",
    # Processing Exceptions
    "FilterError",
    "TransformError",
    "ColorError",
    "CompositeError",
]


class DreamstackRasterError(Exception):
    """Base exception for all dreamstack-raster errors."""


# =============================================================================
# File/Path Exceptions
# =============================================================================


class InvalidPathError(DreamstackRasterError):
    """Raised when a path is invalid or does not exist."""


class PathNotFoundError(DreamstackRasterError):
    """Raised when a file or directory does not exist."""


class NotAFileError(DreamstackRasterError):
    """Raised when a path exists but is not a file."""


class DirectoryNotFoundError(DreamstackRasterError):
    """Raised when a path exists but is not a directory."""


# =============================================================================
# Image Exceptions
# =============================================================================


class InvalidImageError(DreamstackRasterError):
    """Raised when an image is invalid or corrupted."""


class InvalidImageTypeError(DreamstackRasterError):
    """Raised when the image type/format is not supported or invalid."""


class UnsupportedFormatError(DreamstackRasterError):
    """Raised when attempting to use an unsupported image format."""


class ImageProcessingError(DreamstackRasterError):
    """Raised when image processing fails."""


class ImageReadError(DreamstackRasterError):
    """Raised when reading an image fails."""


class ImageWriteError(DreamstackRasterError):
    """Raised when writing an image fails."""


# =============================================================================
# Parameter Exceptions
# =============================================================================


class InvalidParameterError(DreamstackRasterError):
    """Raised when a function receives an invalid parameter value."""


class MutualExclusionError(DreamstackRasterError):
    """Raised when mutually exclusive parameters are both provided."""


class MissingRequiredParameterError(DreamstackRasterError):
    """Raised when a required parameter is missing."""


# =============================================================================
# Processing Exceptions
# =============================================================================


class FilterError(DreamstackRasterError):
    """Raised when a filter operation fails."""


class TransformError(DreamstackRasterError):
    """Raised when a transform operation fails."""


class ColorError(DreamstackRasterError):
    """Raised when a color operation fails."""


class CompositeError(DreamstackRasterError):
    """Raised when a compositing operation fails."""
