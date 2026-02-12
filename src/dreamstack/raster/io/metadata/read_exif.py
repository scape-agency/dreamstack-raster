"""
Dreamstack Raster - Read EXIF Metadata
======================================

Read EXIF metadata from image files.

"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def read_exif(path: str | Path) -> dict[str, Any]:
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
        return _read_exif_pil(Path(path))

    path = Path(path)

    try:
        exif_dict = piexif.load(str(path))
    except (OSError, ValueError):
        return {}

    result = {}

    # IFD names mapping (used for tag lookup)
    _ifd_tags = {  # noqa: F841
        "0th": piexif.TAGS.get(0, {}),  # type: ignore[call-overload]
        "Exif": piexif.TAGS.get(34665, {}),  # type: ignore[call-overload]
        "GPS": piexif.TAGS.get(34853, {}),  # type: ignore[call-overload]
        "1st": piexif.TAGS.get(1, {}),  # type: ignore[call-overload]
    }

    for ifd_name, _tags in [
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


def _read_exif_pil(path: Path) -> dict[str, Any]:
    """Read EXIF using PIL fallback."""
    from PIL import Image as PILImage
    from PIL.ExifTags import TAGS

    result = {}

    with PILImage.open(path) as img:
        exif = img._getexif()  # type: ignore[attr-defined]  # pylint: disable=protected-access
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
