# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Save HDR
============================

Save HDR/Radiance image.

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
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.image import Image


def save_hdr(
    image: Image,
    path: Path,
    **_options,  # noqa: ARG001
) -> None:
    """Save HDR/Radiance image."""
    # pylint: disable=import-outside-toplevel
    import imageio

    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.pixel import BitDepth

    # Convert to float RGB
    rgb_image = image.to_rgb()
    if rgb_image.bit_depth != BitDepth.FLOAT32:
        rgb_image = rgb_image.convert_bit_depth(BitDepth.FLOAT32)

    imageio.imwrite(path, rgb_image.data, format="HDR-FI")  # type: ignore[call-overload]
