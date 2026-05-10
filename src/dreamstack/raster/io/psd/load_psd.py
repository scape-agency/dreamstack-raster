# -*- coding: utf-8 -*-
# pylint: disable=invalid-name


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Load PSD
============================

Load Adobe Photoshop PSD/PSB files.

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


def load_psd(
    path: str | Path,
    layers: bool = False,
    layer_index: int | None = None,
    **_options,  # noqa: ARG001
) -> Image:
    """
    Load a Photoshop PSD/PSB file.

    Args:
        path: Path to PSD file
        layers: If True, return Document with layers preserved
        layer_index: Load only specific layer by index
        **options: Additional options

    Returns:
        Image (flattened) or Document (if layers=True)
    """
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster._optional import require

    PSDImage = require(
        "psd_tools", extra="psd", feature="PSD/PSB loading"
    ).PSDImage

    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.image import Image, ImageMetadata

    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.pixel import BitDepth, PixelData, PixelFormat

    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.io.psd.load_psd_with_layers import (
        load_psd_with_layers,
    )

    path = Path(path)
    psd = PSDImage.open(path)

    if layers:
        return load_psd_with_layers(psd, path.stem)  # type: ignore[return-value]

    if layer_index is not None:
        # Load specific layer
        layer = psd[layer_index]
        pil_image = layer.composite()
    else:
        # Get composite (flattened) image
        pil_image = psd.composite()

    if pil_image is None:
        raise ValueError("Failed to load PSD composite image")

    # Convert to numpy array
    array = np.array(pil_image)

    # Determine format
    if pil_image.mode == "RGBA":
        pixel_format = PixelFormat.RGBA
    elif pil_image.mode == "RGB":
        pixel_format = PixelFormat.RGB
    elif pil_image.mode == "LA":
        pixel_format = PixelFormat.GRAY_ALPHA
    elif pil_image.mode == "L":
        pixel_format = PixelFormat.GRAY
    elif pil_image.mode == "CMYK":
        pixel_format = PixelFormat.CMYK
    else:
        array = np.array(pil_image.convert("RGBA"))
        pixel_format = PixelFormat.RGBA

    # Bit depth
    if array.dtype == np.uint8:
        bit_depth = BitDepth.UINT8
    elif array.dtype == np.uint16:
        bit_depth = BitDepth.UINT16
    else:
        bit_depth = BitDepth.UINT8
        array = array.astype(np.uint8)

    if array.ndim == 2:
        array = array[:, :, np.newaxis]

    pixel_data = PixelData(
        data=array, pixel_format=pixel_format, bit_depth=bit_depth
    )

    # Metadata
    dpi_value = getattr(psd, "dpi", None)
    metadata = ImageMetadata(
        dpi=(dpi_value, dpi_value) if dpi_value else (72.0, 72.0)
    )

    return Image(pixel_data, metadata, name=path.stem)
