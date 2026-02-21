# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Load with PIL
=================================

Load image using PIL.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

# =============================================================================
# Type Checking Imports
# =============================================================================

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.image import Image


def load_with_pil(path: Path, **_options) -> Image:  # noqa: ARG001
    """Load image using PIL."""
    # pylint: disable=import-outside-toplevel
    from PIL import Image as PILImage

    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.image import Image, ImageMetadata

    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.pixel import BitDepth, PixelData, PixelFormat

    pil_image = PILImage.open(path)

    # Handle different modes
    mode = pil_image.mode

    if mode == "P":  # Palette
        pil_image = pil_image.convert("RGBA")
        mode = "RGBA"
    elif mode == "LA":  # Grayscale with alpha
        pass
    elif mode == "L":  # Grayscale
        pass
    elif mode == "1":  # Binary
        pil_image = pil_image.convert("L")
        mode = "L"
    elif mode == "I":  # 32-bit integer
        pil_image = pil_image.convert("I;16")
        mode = "I;16"
    elif mode == "F":  # 32-bit float
        pass
    elif mode == "CMYK":
        pass
    elif mode == "RGB":
        pass
    elif mode == "RGBA":
        pass
    else:
        # Default to RGB
        pil_image = pil_image.convert("RGB")
        mode = "RGB"

    # Convert to numpy array
    array = np.array(pil_image)

    # Determine pixel format and bit depth
    if mode == "L":
        pixel_format = PixelFormat.GRAY
    elif mode == "LA":
        pixel_format = PixelFormat.GRAY_ALPHA
    elif mode == "RGB":
        pixel_format = PixelFormat.RGB
    elif mode == "RGBA":
        pixel_format = PixelFormat.RGBA
    elif mode == "CMYK":
        pixel_format = PixelFormat.CMYK
    else:
        pixel_format = PixelFormat.RGB
        if array.ndim == 2:
            array = np.stack([array, array, array], axis=2)

    # Determine bit depth
    if array.dtype == np.uint8:
        bit_depth = BitDepth.UINT8
    elif array.dtype == np.uint16:
        bit_depth = BitDepth.UINT16
    elif array.dtype == np.float32:
        bit_depth = BitDepth.FLOAT32
    else:
        array = array.astype(np.uint8)
        bit_depth = BitDepth.UINT8

    # Ensure 3D array
    if array.ndim == 2:
        array = array[:, :, np.newaxis]

    # Create pixel data
    pixel_data = PixelData(
        data=array, pixel_format=pixel_format, bit_depth=bit_depth
    )

    # Extract metadata
    metadata = ImageMetadata()

    # Get DPI
    if hasattr(pil_image, "info") and "dpi" in pil_image.info:
        dpi = pil_image.info["dpi"]
        metadata.dpi = (float(dpi[0]), float(dpi[1]))

    # Get EXIF
    try:
        # pylint: disable=import-outside-toplevel
        from PIL.ExifTags import (
            TAGS,
        )  # pylint: disable=import-outside-toplevel

        # pylint: disable=protected-access
        exif = pil_image._getexif()  # type: ignore[union-attr]
        # pylint: enable=protected-access
        if exif:
            metadata.exif = {TAGS.get(k, k): v for k, v in exif.items()}
    except (AttributeError, KeyError):
        pass

    # Get ICC profile
    if "icc_profile" in pil_image.info:
        metadata.custom["icc_profile"] = pil_image.info["icc_profile"]

    return Image(pixel_data, metadata, name=path.stem)
