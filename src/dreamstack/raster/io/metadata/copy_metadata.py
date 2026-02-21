# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Copy Metadata
=================================

Copy metadata between image files.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from pathlib import Path


def copy_metadata(src_path: str | Path, dst_path: str | Path) -> None:
    """
    Copy metadata from one image to another.

    Args:
        src_path: Source image path
        dst_path: Destination image path
    """
    try:
        import piexif  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise ImportError(
            "piexif package required for copying metadata"
        ) from exc

    src_path = Path(src_path)
    dst_path = Path(dst_path)

    try:
        exif_dict = piexif.load(str(src_path))
        exif_bytes = piexif.dump(exif_dict)
        piexif.insert(exif_bytes, str(dst_path))
    except Exception as e:
        raise RuntimeError(f"Failed to copy metadata: {e}") from e
