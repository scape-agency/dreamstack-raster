# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - OpenEXR File Information
=============================================

Get information about EXR files.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from pathlib import Path


def get_exr_info(path: str | Path) -> dict:
    """
    Get information about an EXR file.

    Args:
        path: Path to EXR file

    Returns:
        Dictionary with EXR file information
    """
    try:
        import OpenEXR  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise ImportError(
            "OpenEXR package required for this function"
        ) from exc

    path = Path(path)
    exr_file = OpenEXR.InputFile(str(path))
    header = exr_file.header()

    dw = header["dataWindow"]

    info = {
        "width": dw.max.x - dw.min.x + 1,
        "height": dw.max.y - dw.min.y + 1,
        "channels": list(header["channels"].keys()),
        "compression": str(header.get("compression", "unknown")),
    }

    # Add display window if different
    disp = header.get("displayWindow")
    if disp:
        info["display_width"] = disp.max.x - disp.min.x + 1
        info["display_height"] = disp.max.y - disp.min.y + 1

    # Common attributes
    for attr in ["owner", "comments", "xDensity", "capDate"]:
        if attr in header:
            info[attr] = header[attr]

    return info
