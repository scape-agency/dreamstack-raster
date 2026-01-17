# -*- coding: utf-8 -*-

"""
Dreamstack Raster - RAW Image Loading
=====================================

Load RAW camera files.

"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

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
    **options,
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
        params = rawpy.Params(
            use_camera_wb=use_camera_wb,
            use_auto_wb=use_auto_wb,
            bright=bright,
            no_auto_bright=no_auto_bright,
            gamma=gamma,
            output_bps=output_bps,
        )

        # Output color space
        color_spaces = {
            "sRGB": rawpy.ColorSpace.sRGB,
            "Adobe": rawpy.ColorSpace.Adobe,
            "Wide": rawpy.ColorSpace.Wide,
            "ProPhoto": rawpy.ColorSpace.ProPhoto,
            "XYZ": rawpy.ColorSpace.XYZ,
            "raw": rawpy.ColorSpace.raw,
        }
        params.output_color = color_spaces.get(
            output_color, rawpy.ColorSpace.sRGB
        )

        # Demosaicing algorithm
        demosaic_algorithms = {
            "linear": rawpy.DemosaicAlgorithm.LINEAR,
            "VNG": rawpy.DemosaicAlgorithm.VNG,
            "PPG": rawpy.DemosaicAlgorithm.PPG,
            "AHD": rawpy.DemosaicAlgorithm.AHD,
            "DCB": rawpy.DemosaicAlgorithm.DCB,
            "DHT": rawpy.DemosaicAlgorithm.DHT,
            "AAHD": rawpy.DemosaicAlgorithm.AAHD,
        }
        params.demosaic_algorithm = demosaic_algorithms.get(
            demosaic_algorithm, rawpy.DemosaicAlgorithm.AHD
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
        except Exception:
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
