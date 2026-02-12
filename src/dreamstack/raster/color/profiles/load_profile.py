"""Load ICC profile from file."""

from __future__ import annotations

from pathlib import Path

from dreamstack.raster.color.profiles.icc_profile import ICCProfile


def load_profile(path: str | Path) -> ICCProfile:
    """
    Load ICC profile from file.

    Args:
        path: Path to .icc or .icm file

    Returns:
        ICCProfile
    """
    return ICCProfile.from_file(path)
