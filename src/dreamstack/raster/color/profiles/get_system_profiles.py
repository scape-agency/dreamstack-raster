# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Get system ICC profiles."""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

import platform
from pathlib import Path


def _get_system_profile_paths() -> list[tuple[str, Path]]:
    """Get paths to system ICC profiles."""
    profiles = []
    system = platform.system()

    if system == "Darwin":  # macOS
        paths = [
            Path("/System/Library/ColorSync/Profiles"),
            Path("/Library/ColorSync/Profiles"),
            Path.home() / "Library/ColorSync/Profiles",
        ]
    elif system == "Windows":
        paths = [
            Path("C:/Windows/System32/spool/drivers/color"),
        ]
    else:  # Linux
        paths = [
            Path("/usr/share/color/icc"),
            Path("/usr/local/share/color/icc"),
            Path.home() / ".local/share/icc",
            Path.home() / ".color/icc",
        ]

    for path in paths:
        if path.exists():
            for f in path.glob("*.ic[cm]"):
                profiles.append((f.stem, f))
            for f in path.glob("*.IC[CM]"):
                profiles.append((f.stem, f))

    return profiles


def get_system_profiles() -> dict[str, Path]:
    """
    Get available system ICC profiles.

    Returns:
        Dictionary mapping profile names to paths
    """
    return {name: path for name, path in _get_system_profile_paths()}
