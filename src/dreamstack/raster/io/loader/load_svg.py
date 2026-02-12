"""
Dreamstack Raster - Load SVG
============================

Load and rasterize SVG.

"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dreamstack.raster.core.image import Image


def load_svg(
    path: Path,
    width: int | None = None,
    height: int | None = None,
    dpi: float = 96,
    **_options,  # noqa: ARG001
) -> Image:
    """Load and rasterize SVG."""
    from io import BytesIO

    import cairosvg
    from PIL import Image as PILImage

    from dreamstack.raster.core.image import Image, ImageMetadata
    from dreamstack.raster.core.pixel import BitDepth, PixelData, PixelFormat

    # Rasterize SVG
    png_data = cairosvg.svg2png(
        url=str(path), output_width=width, output_height=height, dpi=int(dpi)
    )

    # Load PNG data
    pil_image = PILImage.open(BytesIO(png_data))  # type: ignore[arg-type]
    array = np.array(pil_image)

    pixel_data = PixelData(
        data=array, pixel_format=PixelFormat.RGBA, bit_depth=BitDepth.UINT8
    )

    metadata = ImageMetadata(dpi=(dpi, dpi))

    return Image(pixel_data, metadata, name=path.stem)
