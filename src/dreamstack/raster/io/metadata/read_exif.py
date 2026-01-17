# -*- coding: utf-8 -*-

"""
Dreamstack Raster - Read EXIF Metadata
======================================

Read EXIF metadata from image files.

"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def read_exif(path: str | Path) -> Dict[str, Any]:
    """
    Read EXIF metadata from an image.

    Args:
        path: Path to image file

    Returns:
        Dictionary of EXIF tags
    """
    try:
        import piexif
    except ImportError:
        return _read_exif_pil(path)

    path = Path(path)

    try:
        exif_dict = piexif.load(str(path))
    except Exception:
        return {}

    result = {}

    # IFD names
    ifd_names = {
        "0th": piexif.TAGS.get(0, {}),
        "Exif": piexif.TAGS.get(34665, {}),
        "GPS": piexif.TAGS.get(34853, {}),
        "1st": piexif.TAGS.get(1, {}),
    }

    for ifd_name, tags in [
        ("0th", piexif.ImageIFD),
        ("Exif", piexif.ExifIFD),
        ("GPS", piexif.GPSIFD),
    ]:
        ifd_data = exif_dict.get(ifd_name, {})
        for tag, value in ifd_data.items():
            tag_name = (
                piexif.TAGS.get(ifd_name, {})
                .get(tag, {})
                .get("name", str(tag))
            )

            # Convert bytes to string
            if isinstance(value, bytes):
                try:
                    value = value.decode("utf-8").strip("\x00")
                except UnicodeDecodeError:
                    value = value.hex()

            result[tag_name] = value

    return result


def _read_exif_pil(path: Path) -> Dict[str, Any]:
    """Read EXIF using PIL fallback."""
    from PIL import Image as PILImage
    from PIL.ExifTags import GPSTAGS, TAGS

    result = {}

    with PILImage.open(path) as img:
        exif = img._getexif()
        if exif:
            for tag_id, value in exif.items():
                tag = TAGS.get(tag_id, tag_id)

                # Handle bytes
                if isinstance(value, bytes):
                    try:
                        value = value.decode("utf-8").strip("\x00")
                    except UnicodeDecodeError:
                        continue

                result[tag] = value

    return result
