# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - OpenEXR Loading
===================================

Load OpenEXR image files.

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


def load_exr(
    path: str | Path, channels: list[str] | None = None, **options
) -> Image:
    """
    Load an OpenEXR image.

    Args:
        path: Path to EXR file
        channels: Specific channels to load (e.g., ['R', 'G', 'B', 'A'])
        **options: Additional options

    Returns:
        Loaded Image with float data
    """
    # pylint: disable=import-outside-toplevel,c-extension-no-member
    try:
        import Imath
        import OpenEXR
    except ImportError:
        # Fallback to imageio
        return _load_exr_imageio(Path(path), **options)

    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.image import Image, ImageMetadata

    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.pixel import BitDepth, PixelData, PixelFormat

    path = Path(path)
    exr_file = OpenEXR.InputFile(str(path))

    # Get header info
    header = exr_file.header()
    dw = header["dataWindow"]
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    # Get available channels
    available_channels = list(header["channels"].keys())

    # Determine which channels to load
    if channels is None:
        # Try to detect standard channel configurations
        if all(c in available_channels for c in ["R", "G", "B", "A"]):
            channels = ["R", "G", "B", "A"]
            pixel_format = PixelFormat.RGBA
        elif all(c in available_channels for c in ["R", "G", "B"]):
            channels = ["R", "G", "B"]
            pixel_format = PixelFormat.RGB
        elif "Y" in available_channels:
            channels = ["Y"]
            pixel_format = PixelFormat.GRAY
        else:
            channels = available_channels[:4]  # Take first 4 channels
            pixel_format = (
                PixelFormat.RGBA if len(channels) == 4 else PixelFormat.RGB
            )
    else:
        if len(channels) == 4:
            pixel_format = PixelFormat.RGBA
        elif len(channels) == 3:
            pixel_format = PixelFormat.RGB
        else:
            pixel_format = PixelFormat.GRAY

    # Determine pixel type
    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    # Read channels
    channel_data = []
    for channel_name in channels:
        if channel_name in available_channels:
            raw_data = exr_file.channel(channel_name, pt)
            channel_array = np.frombuffer(raw_data, dtype=np.float32)
            channel_array = channel_array.reshape((height, width))
            channel_data.append(channel_array)
        else:
            # Fill with zeros if channel not present
            channel_data.append(np.zeros((height, width), dtype=np.float32))

    # Stack channels
    if len(channel_data) == 1:
        data = channel_data[0][:, :, np.newaxis]
    else:
        data = np.stack(channel_data, axis=2)

    pixel_data = PixelData(
        data=data, pixel_format=pixel_format, bit_depth=BitDepth.FLOAT32
    )

    # Extract metadata
    metadata = ImageMetadata()

    # Common EXR attributes
    if "xDensity" in header:
        dpi = header["xDensity"] * 25.4  # pixelsPerCm to DPI
        metadata.dpi = (dpi, dpi)

    if "owner" in header:
        metadata.author = header["owner"]

    if "comments" in header:
        metadata.description = header["comments"]

    # Store all header attributes in custom metadata
    for key, value in header.items():
        if isinstance(value, (str, int, float, bool)):
            metadata.custom[key] = value

    return Image(pixel_data, metadata, name=path.stem)


def _load_exr_imageio(path: Path, **_options) -> Image:
    """Fallback EXR loading using imageio."""
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster._optional import require

    imageio = require("imageio", extra="exr", feature="EXR loading")

    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.image import Image, ImageMetadata

    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.pixel import BitDepth, PixelData, PixelFormat

    data = imageio.imread(path, format="EXR-FI")

    if data.ndim == 2:
        pixel_format = PixelFormat.GRAY
        data = data[:, :, np.newaxis]
    elif data.shape[2] == 3:
        pixel_format = PixelFormat.RGB
    else:
        pixel_format = PixelFormat.RGBA

    pixel_data = PixelData(
        data=data.astype(np.float32),
        pixel_format=pixel_format,
        bit_depth=BitDepth.FLOAT32,
    )

    return Image(pixel_data, ImageMetadata(), name=path.stem)
