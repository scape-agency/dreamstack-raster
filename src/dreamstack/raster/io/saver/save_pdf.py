"""
Dreamstack Raster - Save PDF
============================

Save image as PDF.

"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dreamstack.raster.core.image import Image


def save_pdf(image: Image, path: Path, **options) -> None:
    """Save image as PDF."""
    pil_image = image.to_pil()

    # Convert to RGB if needed
    if pil_image.mode == "RGBA":
        # pylint: disable=import-outside-toplevel
        from PIL import Image as PILImage

        background = PILImage.new("RGB", pil_image.size, (255, 255, 255))
        background.paste(pil_image, mask=pil_image.split()[3])
        pil_image = background
    elif pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    # Calculate PDF size (kept for potential future use)
    dpi = image.metadata.dpi
    _width_inches = image.width / dpi[0]
    _height_inches = image.height / dpi[1]

    pil_image.save(
        path,
        "PDF",
        resolution=dpi[0],
        title=options.get("title", image.name),
        author=options.get("author", image.metadata.author),
    )
