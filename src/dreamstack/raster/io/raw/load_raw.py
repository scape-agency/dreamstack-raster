# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - RAW Image Loading
=====================================

Load RAW camera files.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dreamstack.raster.core.image import Image


def load_raw(
    path: str | Path,
    use_camera_wb: bool = True,
    use_auto_wb: bool = False,
    bright: float = 1.0,
    output_color: str = "sRGB",
    output_bps: int = 16,
    no_auto_bright: bool = False,
    gamma: tuple = (1, 1),
    demosaic_algorithm: str = "AHD",
    **_options,  # noqa: ARG001
) -> Image:
    """
    Load a RAW camera file.

    Args:
        path: Path to RAW file
        use_camera_wb: Use camera white balance
        use_auto_wb: Calculate automatic white balance
        bright: Brightness adjustment factor
        output_color: Output color space (sRGB, Adobe, Wide, ProPhoto, XYZ)
        output_bps: Output bits per sample (8 or 16)
        no_auto_bright: Disable automatic brightness adjustment
        gamma: Gamma curve (power, slope)
        demosaic_algorithm: Demosaicing algorithm
        **options: Additional rawpy options

    Returns:
        Processed Image
    """
    import rawpy

    from dreamstack.raster.core.image import Image, ImageMetadata
    from dreamstack.raster.core.pixel import BitDepth, PixelData, PixelFormat

    path = Path(path)

    with rawpy.imread(str(path)) as raw:
        # Configure postprocessing
        Params = getattr(rawpy, "Params")
        params = Params(
            use_camera_wb=use_camera_wb,
            use_auto_wb=use_auto_wb,
            bright=bright,
            no_auto_bright=no_auto_bright,
            gamma=gamma,
            output_bps=output_bps,
        )

        # Output color space
        ColorSpace = getattr(rawpy, "ColorSpace")
        color_spaces = {
            "sRGB": ColorSpace.sRGB,
            "Adobe": ColorSpace.Adobe,
            "Wide": ColorSpace.Wide,
            "ProPhoto": ColorSpace.ProPhoto,
            "XYZ": ColorSpace.XYZ,
            "raw": ColorSpace.raw,
        }
        params.output_color = color_spaces.get(output_color, ColorSpace.sRGB)

        # Demosaicing algorithm
        DemosaicAlgorithm = getattr(rawpy, "DemosaicAlgorithm")
        demosaic_algorithms = {
            "linear": DemosaicAlgorithm.LINEAR,
            "VNG": DemosaicAlgorithm.VNG,
            "PPG": DemosaicAlgorithm.PPG,
            "AHD": DemosaicAlgorithm.AHD,
            "DCB": DemosaicAlgorithm.DCB,
            "DHT": DemosaicAlgorithm.DHT,
            "AAHD": DemosaicAlgorithm.AAHD,
        }
        params.demosaic_algorithm = demosaic_algorithms.get(
            demosaic_algorithm, DemosaicAlgorithm.AHD
        )

        # Process
        rgb = raw.postprocess(params)

        # Extract metadata
        metadata = ImageMetadata()

        # Get EXIF from raw
        try:
            if hasattr(raw, "metadata"):
                metadata.exif = {
                    "iso": (
                        raw.metadata.iso_speed
                        if hasattr(raw.metadata, "iso_speed")
                        else None
                    ),
                    "shutter": (
                        raw.metadata.shutter
                        if hasattr(raw.metadata, "shutter")
                        else None
                    ),
                    "aperture": (
                        raw.metadata.aperture
                        if hasattr(raw.metadata, "aperture")
                        else None
                    ),
                    "focal_length": (
                        raw.metadata.focal_len
                        if hasattr(raw.metadata, "focal_len")
                        else None
                    ),
                    "camera": (
                        f"{raw.metadata.make} {raw.metadata.model}"
                        if hasattr(raw.metadata, "make")
                        else None
                    ),
                }
        except (AttributeError, KeyError, OSError):
            pass

    # Determine bit depth
    if output_bps == 16:
        bit_depth = BitDepth.UINT16
    else:
        bit_depth = BitDepth.UINT8

    pixel_data = PixelData(
        data=rgb, pixel_format=PixelFormat.RGB, bit_depth=bit_depth
    )

    metadata.color_profile = output_color

    return Image(pixel_data, metadata, name=path.stem)
