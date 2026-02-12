"""
Dreamstack Raster - Load Image Info
===================================

Load only image metadata without full pixel data.

"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_image_info(path: str | Path) -> dict[str, Any]:
    """
    Load only image metadata without full pixel data.

    Args:
        path: Image path

    Returns:
        Dictionary with image information
    """
    from PIL import Image as PILImage

    path = Path(path)

    with PILImage.open(path) as img:
        info = {
            "width": img.width,
            "height": img.height,
            "mode": img.mode,
            "format": img.format,
            "format_description": (
                img.format_description
                if hasattr(img, "format_description")
                else None
            ),
        }

        if "dpi" in img.info:
            info["dpi"] = img.info["dpi"]

        if hasattr(img, "n_frames"):
            info["frames"] = img.n_frames  # type: ignore[attr-defined]

        # Size info
        info["megapixels"] = (img.width * img.height) / 1_000_000

        # File size
        info["file_size"] = path.stat().st_size

        return info
