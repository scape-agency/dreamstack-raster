# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Write Metadata
==================================

Write metadata to image files.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from pathlib import Path
from typing import Any


def write_metadata(
    path: str | Path, metadata: dict[str, Any], preserve_existing: bool = True
) -> None:
    """
    Write metadata to an image file.

    Args:
        path: Path to image file
        metadata: Metadata to write
        preserve_existing: Whether to preserve existing metadata
    """
    try:
        import piexif  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise ImportError(
            "piexif package required for writing metadata"
        ) from exc

    path = Path(path)

    # Load existing EXIF
    if preserve_existing:
        try:
            exif_dict = piexif.load(str(path))
        except (OSError, ValueError, KeyError):  # piexif parsing errors
            exif_dict = {
                "0th": {},
                "Exif": {},
                "GPS": {},
                "1st": {},
                "thumbnail": None,
            }
    else:
        exif_dict = {
            "0th": {},
            "Exif": {},
            "GPS": {},
            "1st": {},
            "thumbnail": None,
        }

    # Map common metadata keys to EXIF tags
    mappings = {
        "author": (piexif.ImageIFD.Artist, "0th"),
        "copyright": (piexif.ImageIFD.Copyright, "0th"),
        "description": (piexif.ImageIFD.ImageDescription, "0th"),
        "software": (piexif.ImageIFD.Software, "0th"),
        "datetime": (piexif.ImageIFD.DateTime, "0th"),
        "make": (piexif.ImageIFD.Make, "0th"),
        "model": (piexif.ImageIFD.Model, "0th"),
    }

    for key, value in metadata.items():
        if key.lower() in mappings:
            tag, ifd = mappings[key.lower()]
            if isinstance(value, str):
                value = value.encode("utf-8")
            exif_dict[ifd][tag] = value

    # Write back
    exif_bytes = piexif.dump(exif_dict)
    piexif.insert(exif_bytes, str(path))
