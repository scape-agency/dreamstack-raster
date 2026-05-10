# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - RAW Thumbnail Extraction
============================================

Extract embedded thumbnails from RAW camera files.

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


def get_raw_thumbnail(path: str | Path) -> Image:
    """
    Extract embedded thumbnail from RAW file.

    Args:
        path: Path to RAW file

    Returns:
        Thumbnail Image
    """
    # pylint: disable=import-outside-toplevel
    from io import BytesIO

    # pylint: disable=import-outside-toplevel
    from dreamstack.raster._optional import require

    rawpy = require("rawpy", extra="raw", feature="RAW thumbnail extraction")

    # pylint: disable=import-outside-toplevel
    from PIL import Image as PILImage

    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.image import Image, ImageMetadata

    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.pixel import BitDepth, PixelData, PixelFormat

    path = Path(path)

    with rawpy.imread(str(path)) as raw:
        try:
            thumb = raw.extract_thumb()

            if thumb.format == getattr(rawpy, "ThumbFormat").JPEG:
                pil_image = PILImage.open(BytesIO(thumb.data))
            elif thumb.format == getattr(rawpy, "ThumbFormat").BITMAP:
                # Raw RGB data
                pil_image = PILImage.frombytes(
                    "RGB", (thumb.width, thumb.height), thumb.data
                )
            else:
                raise ValueError(f"Unknown thumbnail format: {thumb.format}")

            array = np.array(pil_image)

            if array.ndim == 2:
                array = np.stack([array, array, array], axis=2)

            pixel_data = PixelData(
                data=array,
                pixel_format=PixelFormat.RGB,
                bit_depth=BitDepth.UINT8,
            )

            return Image(
                pixel_data, ImageMetadata(), name=f"{path.stem}_thumb"
            )

        except Exception as e:
            raise ValueError(f"No thumbnail available: {e}") from e
