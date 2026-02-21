# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Read Metadata
=================================

Read metadata from image files.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from pathlib import Path
from typing import Any

from dreamstack.raster.io.metadata.read_exif import read_exif
from dreamstack.raster.io.metadata.read_iptc import read_iptc
from dreamstack.raster.io.metadata.read_xmp import read_xmp


def read_metadata(path: str | Path) -> dict[str, Any]:
    """
    Read metadata from an image file.

    Args:
        path: Path to image file

    Returns:
        Dictionary of metadata
    """
    # pylint: disable=import-outside-toplevel
    from PIL import Image as PILImage

    path = Path(path)
    metadata = {}

    with PILImage.open(path) as img:
        # Basic info
        metadata["width"] = img.width
        metadata["height"] = img.height
        metadata["mode"] = img.mode
        metadata["format"] = img.format

        # PIL info dict
        for key, value in img.info.items():
            if isinstance(value, (str, int, float, bool, tuple, list)):
                metadata[key] = value

        # EXIF data
        try:
            exif = read_exif(path)
            if exif:
                metadata["exif"] = exif
        except Exception:
            pass

        # XMP data
        try:
            xmp = read_xmp(path)
            if xmp:
                metadata["xmp"] = xmp
        except Exception:
            pass

        # IPTC data
        try:
            iptc = read_iptc(path)
            if iptc:
                metadata["iptc"] = iptc
        except Exception:
            pass

    return metadata
