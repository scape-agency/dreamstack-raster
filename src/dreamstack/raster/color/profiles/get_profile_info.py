# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Get detailed information about an ICC profile."""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from typing import TYPE_CHECKING, Any

# =============================================================================
# Type Checking Imports
# =============================================================================

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.color.profiles.icc_profile import ICCProfile


def get_profile_info(profile: ICCProfile) -> dict[str, Any]:
    """
    Get detailed information about an ICC profile.

    Args:
        profile: ICC profile

    Returns:
        Dictionary with profile information
    """
    return {
        "name": profile.name,
        "version": profile.version,
        "class": profile.profile_class.name if profile.profile_class else None,
        "color_space": (
            profile.color_space.name if profile.color_space else None
        ),
        "pcs": profile.pcs,
        "rendering_intent": profile.rendering_intent.name,
        "copyright": profile.copyright,
        "size": len(profile.data),
    }
