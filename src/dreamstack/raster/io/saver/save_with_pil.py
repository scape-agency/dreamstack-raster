# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Save with PIL
=================================

Save image using PIL.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from dreamstack.raster.io.formats import ImageFormat

# =============================================================================
# Type Checking Imports
# =============================================================================

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.image import Image


def save_with_pil(
    image: Image, path: Path, image_format: ImageFormat, **options
) -> None:
    """Save image using PIL."""
    # pylint: disable=import-outside-toplevel
    from PIL import Image as PILImage

    pil_image = image.to_pil()

    # Format-specific handling
    save_kwargs = {}

    if image_format == ImageFormat.PNG:
        save_kwargs["format"] = "PNG"
        save_kwargs["compress_level"] = options.get("compression", 6)
        if options.get("optimize", False):
            save_kwargs["optimize"] = True

    elif image_format == ImageFormat.JPEG:
        save_kwargs["format"] = "JPEG"
        save_kwargs["quality"] = options.get("quality", 95)
        save_kwargs["optimize"] = options.get("optimize", True)
        if options.get("progressive", False):
            save_kwargs["progressive"] = True

        # JPEG doesn't support alpha - convert if needed
        if pil_image.mode in ("RGBA", "LA"):
            background = PILImage.new("RGB", pil_image.size, (255, 255, 255))
            if pil_image.mode == "RGBA":
                background.paste(pil_image, mask=pil_image.split()[3])
            else:
                background.paste(pil_image, mask=pil_image.split()[1])
            pil_image = background

    elif image_format == ImageFormat.TIFF:
        save_kwargs["format"] = "TIFF"
        compression_map = {
            "none": "raw",
            "lzw": "tiff_lzw",
            "zip": "tiff_deflate",
            "jpeg": "jpeg",
        }
        compression = options.get("compression", "lzw")
        save_kwargs["compression"] = compression_map.get(
            compression, "tiff_lzw"
        )

    elif image_format == ImageFormat.WEBP:
        save_kwargs["format"] = "WEBP"
        if options.get("lossless", False):
            save_kwargs["lossless"] = True
        else:
            save_kwargs["quality"] = options.get("quality", 80)

    elif image_format == ImageFormat.GIF:
        save_kwargs["format"] = "GIF"
        # Convert to palette mode
        if pil_image.mode != "P":
            pil_image = pil_image.convert(
                "P", palette=getattr(PILImage, "ADAPTIVE"), colors=256
            )

    elif image_format == ImageFormat.BMP:
        save_kwargs["format"] = "BMP"

    elif image_format == ImageFormat.ICO:
        save_kwargs["format"] = "ICO"
        # ICO typically needs specific sizes
        sizes = options.get(
            "sizes", [(256, 256), (128, 128), (64, 64), (32, 32), (16, 16)]
        )
        save_kwargs["sizes"] = sizes

    elif image_format in (ImageFormat.HEIC, ImageFormat.HEIF):
        # pylint: disable=import-outside-toplevel
        from pillow_heif import register_heif_opener

        register_heif_opener()
        save_kwargs["format"] = "HEIF"
        save_kwargs["quality"] = options.get("quality", 80)

    elif image_format == ImageFormat.AVIF:
        save_kwargs["format"] = "AVIF"
        save_kwargs["quality"] = options.get("quality", 80)

    # Add DPI
    dpi = image.metadata.dpi
    save_kwargs["dpi"] = dpi

    # Add EXIF if present
    if image.metadata.exif and image_format in (
        ImageFormat.JPEG,
        ImageFormat.TIFF,
    ):
        try:
            import piexif

            exif_dict = {
                "0th": {},
                "Exif": {},
                "GPS": {},
                "1st": {},
                "thumbnail": None,
            }
            # Copy basic EXIF tags
            # This is simplified - full implementation would map all tags
            exif_bytes = piexif.dump(exif_dict)
            save_kwargs["exif"] = exif_bytes
        except (ImportError, KeyError, TypeError):
            pass

    # Add ICC profile
    if "icc_profile" in image.metadata.custom:
        save_kwargs["icc_profile"] = image.metadata.custom["icc_profile"]

    pil_image.save(path, **save_kwargs)
