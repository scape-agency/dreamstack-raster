"""
Dreamstack Raster - OpenEXR Saving
==================================

Save images as OpenEXR files.

"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dreamstack.raster.core.image import Image


def save_exr(
    image: Image,
    path: str | Path,
    compression: str = "zip",
    half_float: bool = False,
    channel_names: list[str] | None = None,
    **options,
) -> None:
    """
    Save an image as OpenEXR.

    Args:
        image: Image to save
        path: Output path
        compression: Compression method (none, rle, zip, zips, piz, pxr24, b44, b44a, dwaa, dwab)
        half_float: Use 16-bit half float instead of 32-bit float
        channel_names: Custom channel names
        **options: Additional options
    """
    try:
        import Imath
        import OpenEXR
    except ImportError:
        _save_exr_imageio(image, Path(path), **options)
        return

    from dreamstack.raster.core.pixel import BitDepth

    path = Path(path)

    # Convert to float if needed
    if image.bit_depth not in (BitDepth.FLOAT16, BitDepth.FLOAT32):
        image = image.convert_bit_depth(BitDepth.FLOAT32)

    # Get dimensions
    height, width = image.height, image.width

    # Determine channel names
    if channel_names is None:
        if image.channels == 1:
            channel_names = ["Y"]
        elif image.channels == 3:
            channel_names = ["R", "G", "B"]
        else:
            channel_names = ["R", "G", "B", "A"]

    # Compression types
    compression_types = {
        "none": (
            OpenEXR.NONE_COMPRESSION
            if hasattr(OpenEXR, "NONE_COMPRESSION")
            else Imath.Compression.NO_COMPRESSION
        ),
        "rle": Imath.Compression.RLE_COMPRESSION,
        "zip": Imath.Compression.ZIP_COMPRESSION,
        "zips": Imath.Compression.ZIPS_COMPRESSION,
        "piz": Imath.Compression.PIZ_COMPRESSION,
        "pxr24": Imath.Compression.PXR24_COMPRESSION,
        "b44": Imath.Compression.B44_COMPRESSION,
        "b44a": Imath.Compression.B44A_COMPRESSION,
    }

    # Create header
    header = OpenEXR.Header(width, height)

    # Set compression
    comp = compression_types.get(compression.lower())
    if comp is not None:
        header["compression"] = comp

    # Set pixel type
    pixel_type = Imath.PixelType.HALF if half_float else Imath.PixelType.FLOAT

    # Define channels
    channel_def = {}
    for name in channel_names:
        channel_def[name] = Imath.Channel(pixel_type)  # type: ignore[arg-type]
    header["channels"] = channel_def

    # Set metadata
    if image.metadata.dpi != (72.0, 72.0):
        header["xDensity"] = image.metadata.dpi[0] / 25.4  # DPI to pixels/cm

    if image.metadata.author:
        header["owner"] = image.metadata.author

    if image.metadata.description:
        header["comments"] = image.metadata.description

    # Create output file
    exr_file = OpenEXR.OutputFile(str(path), header)

    # Prepare channel data
    channel_data = {}
    data = image.data

    if half_float:
        dtype = np.float16
    else:
        dtype = np.float32

    for i, name in enumerate(channel_names[: image.channels]):
        channel_array = data[:, :, i].astype(dtype)
        channel_data[name] = channel_array.tobytes()

    # Write
    exr_file.writePixels(channel_data)
    exr_file.close()


def _save_exr_imageio(image: Image, path: Path, **_options) -> None:
    """Fallback EXR saving using imageio."""
    import imageio

    from dreamstack.raster.core.pixel import BitDepth

    # Convert to float
    if image.bit_depth not in (BitDepth.FLOAT16, BitDepth.FLOAT32):
        image = image.convert_bit_depth(BitDepth.FLOAT32)

    imageio.imwrite(path, image.data.astype(np.float32), format="EXR-FI")  # type: ignore[call-overload]
