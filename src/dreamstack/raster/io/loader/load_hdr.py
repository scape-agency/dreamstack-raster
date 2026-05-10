# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Load HDR
============================

Load HDR/Radiance image.

"""

# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

# =============================================================================
# Type Checking Imports
# =============================================================================

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.image import Image


def load_hdr(
    path: Path,
    **options,  # pylint: disable=unused-argument
) -> Image:
    """Load HDR/Radiance image."""
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster._optional import require

    imageio = require("imageio", extra="exr", feature="HDR/Radiance loading")

    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.image import Image, ImageMetadata

    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.pixel import BitDepth, PixelData, PixelFormat

    data = imageio.imread(path, format="HDR-FI")

    pixel_data = PixelData(
        data=data.astype(np.float32),
        pixel_format=PixelFormat.RGB,
        bit_depth=BitDepth.FLOAT32,
    )

    return Image(pixel_data, ImageMetadata(), name=path.stem)
