# -*- coding: utf-8 -*-

"""
Dreamstack Raster - OpenEXR Channel Information
================================================

Get information about channels in EXR files.

"""

from __future__ import annotations

from pathlib import Path
from typing import Dict


def get_exr_channels(path: str | Path) -> Dict[str, dict]:
    """
    Get information about channels in an EXR file.

    Args:
        path: Path to EXR file

    Returns:
        Dictionary of channel names to channel info
    """
    try:
        import OpenEXR
    except ImportError:
        raise ImportError("OpenEXR package required for this function")

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
