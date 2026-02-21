# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - OpenEXR Channel Information
================================================

Get information about channels in EXR files.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from pathlib import Path


def get_exr_channels(path: str | Path) -> dict[str, dict]:
    """
    Get information about channels in an EXR file.

    Args:
        path: Path to EXR file

    Returns:
        Dictionary of channel names to channel info
    """
    try:
        # pylint: disable=import-outside-toplevel
        import OpenEXR
    except ImportError as exc:
        raise ImportError(
            "OpenEXR package required for this function"
        ) from exc

    path = Path(path)
    exr_file = OpenEXR.InputFile(str(path))
    header = exr_file.header()

    channels = {}
    for name, channel in header["channels"].items():
        channels[name] = {
            "type": str(channel.type),
            "x_sampling": channel.xSampling,
            "y_sampling": channel.ySampling,
        }

    return channels
