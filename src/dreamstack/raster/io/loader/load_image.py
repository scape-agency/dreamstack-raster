# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Load Image
==============================

Universal image loading with format auto-detection.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from dreamstack.raster.io.formats import ImageFormat, get_format_for_path

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.image import Image


def load_image(
    path: str | Path,
    image_format: ImageFormat | None = None,
    **options,
) -> Image:
    """
    Load an image from file.

    Args:
        path: Path to image file
        image_format: Explicit format (auto-detected if None)
        **options: Format-specific loading options

    Returns:
        Loaded Image

    Options by format:
        PNG/JPEG/etc: Standard PIL options
        PSD: layers=True to preserve layers
        RAW: Various rawpy options
        EXR: Various OpenEXR options
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    # Detect format
    if image_format is None:
        image_format = get_format_for_path(path)
        if image_format is None:
            raise ValueError(f"Unknown image format: {path.suffix}")

    # Load based on format
    if image_format == ImageFormat.PSD or image_format == ImageFormat.PSB:
        # pylint: disable=import-outside-toplevel
        from dreamstack.raster.io.psd import load_psd

        return load_psd(path, **options)

    elif image_format in (
        ImageFormat.RAW,
        ImageFormat.CR2,
        ImageFormat.CR3,
        ImageFormat.NEF,
        ImageFormat.ARW,
        ImageFormat.DNG,
        ImageFormat.ORF,
        ImageFormat.RW2,
    ):
        # pylint: disable=import-outside-toplevel
        from dreamstack.raster.io.raw import load_raw

        return load_raw(path, **options)

    elif image_format == ImageFormat.EXR:
        # pylint: disable=import-outside-toplevel
        from dreamstack.raster.io.exr import load_exr

        return load_exr(path, **options)

    elif image_format == ImageFormat.HDR:
        # pylint: disable=import-outside-toplevel
        from dreamstack.raster.io.loader.load_hdr import load_hdr

        return load_hdr(path, **options)

    elif image_format == ImageFormat.SVG:
        # pylint: disable=import-outside-toplevel
        from dreamstack.raster.io.loader.load_svg import load_svg

        return load_svg(path, **options)

    elif image_format in (ImageFormat.HEIC, ImageFormat.HEIF):
        # pylint: disable=import-outside-toplevel
        from dreamstack.raster.io.loader.load_heif import load_heif

        return load_heif(path, **options)

    else:
        # Use PIL for standard formats
        # pylint: disable=import-outside-toplevel
        from dreamstack.raster.io.loader.load_with_pil import load_with_pil

        return load_with_pil(path, **options)
