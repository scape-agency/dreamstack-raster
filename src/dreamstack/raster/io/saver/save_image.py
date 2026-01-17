# -*- coding: utf-8 -*-

"""
Dreamstack Raster - Save Image
==============================

Universal image saving with format support.

"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from dreamstack.raster.io.formats import ImageFormat, get_format_for_path

if TYPE_CHECKING:
    from dreamstack.raster.core.image import Image


def save_image(
    image: Image,
    path: str | Path,
    format: ImageFormat | None = None,
    **options,
) -> None:
    """
    Save an image to file.

    Args:
        image: Image to save
        path: Output path
        format: Explicit format (auto-detected if None)
        **options: Format-specific options

    Options by format:
        PNG:
            compression: 0-9 (default 6)
            optimize: bool

        JPEG:
            quality: 1-100 (default 95)
            progressive: bool
            optimize: bool

        TIFF:
            compression: 'none', 'lzw', 'zip', 'jpeg'

        WEBP:
            quality: 1-100 (default 80)
            lossless: bool

        PSD:
            layers: List of layers

        EXR:
            compression: 'none', 'zip', 'piz', 'pxr24'
    """
    path = Path(path)

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Detect format
    if format is None:
        format = get_format_for_path(path)
        if format is None:
            raise ValueError(f"Unknown image format: {path.suffix}")

    # Save based on format
    if format == ImageFormat.PSD:
        from dreamstack.raster.io.psd import save_psd

        save_psd(image, path, **options)

    elif format == ImageFormat.EXR:
        from dreamstack.raster.io.exr import save_exr

        save_exr(image, path, **options)

    elif format == ImageFormat.HDR:
        from dreamstack.raster.io.saver.save_hdr import save_hdr

        save_hdr(image, path, **options)

    elif format == ImageFormat.PDF:
        from dreamstack.raster.io.saver.save_pdf import save_pdf

        save_pdf(image, path, **options)

    else:
        # Use PIL for standard formats
        from dreamstack.raster.io.saver.save_with_pil import save_with_pil

        save_with_pil(image, path, format, **options)
