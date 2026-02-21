# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""Embed ICC profile in image."""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from typing import TYPE_CHECKING

# =============================================================================
# Type Checking Imports
# =============================================================================

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.color.profiles.icc_profile import ICCProfile

    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.image import Image


def embed_profile(image: Image, profile: ICCProfile) -> Image:
    """
    Embed ICC profile in image.

    Args:
        image: Image to modify
        profile: ICC profile to embed

    Returns:
        Image with embedded profile
    """
    image.metadata.custom["icc_profile"] = profile.data
    image.metadata.color_profile = profile.name
    return image
