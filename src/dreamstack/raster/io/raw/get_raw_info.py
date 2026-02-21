# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - RAW File Information
========================================

Get information about RAW camera files.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

# =============================================================================
# Type Checking Imports
# =============================================================================

if TYPE_CHECKING:
    pass


def get_raw_info(path: str | Path) -> dict:
    """
    Get information about a RAW file without full processing.

    Args:
        path: Path to RAW file

    Returns:
        Dictionary with RAW file information
    """
    # pylint: disable=import-outside-toplevel
    import rawpy

    path = Path(path)

    with rawpy.imread(str(path)) as raw:
        info = {
            "width": raw.sizes.width,
            "height": raw.sizes.height,
            "raw_width": raw.sizes.raw_width,
            "raw_height": raw.sizes.raw_height,
            "flip": raw.sizes.flip,
            "top_margin": raw.sizes.top_margin,
            "left_margin": raw.sizes.left_margin,
        }

        if hasattr(raw, "metadata"):
            meta = raw.metadata
            info["camera_make"] = getattr(meta, "make", None)
            info["camera_model"] = getattr(meta, "model", None)
            info["iso"] = getattr(meta, "iso_speed", None)
            info["shutter"] = getattr(meta, "shutter", None)
            info["aperture"] = getattr(meta, "aperture", None)
            info["focal_length"] = getattr(meta, "focal_len", None)
            info["timestamp"] = getattr(meta, "timestamp", None)

        # Color info
        info["color_desc"] = (
            raw.color_desc.decode() if hasattr(raw, "color_desc") else None
        )
        info["num_colors"] = (
            raw.num_colors if hasattr(raw, "num_colors") else None
        )
        info["raw_pattern"] = (
            raw.raw_pattern.tolist() if hasattr(raw, "raw_pattern") else None
        )

        # White balance
        info["camera_whitebalance"] = (
            raw.camera_whitebalance.tolist()
            if hasattr(raw, "camera_whitebalance")
            else None
        )
        info["daylight_whitebalance"] = (
            raw.daylight_whitebalance.tolist()
            if hasattr(raw, "daylight_whitebalance")
            else None
        )

        return info
