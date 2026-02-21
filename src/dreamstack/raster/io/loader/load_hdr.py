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

if TYPE_CHECKING:
    from dreamstack.raster.core.image import Image


def load_hdr(
    path: Path,
    **options,  # pylint: disable=unused-argument
) -> Image:
    """Load HDR/Radiance image."""
    import imageio

    from dreamstack.raster.core.image import Image, ImageMetadata
    from dreamstack.raster.core.pixel import BitDepth, PixelData, PixelFormat

    data = imageio.imread(path, format="HDR-FI")

    pixel_data = PixelData(
        data=data.astype(np.float32),
        pixel_format=PixelFormat.RGB,
        bit_depth=BitDepth.FLOAT32,
    )

    return Image(pixel_data, ImageMetadata(), name=path.stem)
