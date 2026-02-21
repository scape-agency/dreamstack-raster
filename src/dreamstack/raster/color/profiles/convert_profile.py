# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Convert image from one ICC profile to another."""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from dreamstack.raster.color.profiles.rendering_intent import RenderingIntent

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.color.profiles.icc_profile import ICCProfile

    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.image import Image


def convert_profile(
    image: Image,
    target_profile: ICCProfile,
    source_profile: ICCProfile | None = None,
    intent: RenderingIntent = RenderingIntent.PERCEPTUAL,
) -> Image:
    """
    Convert image from one ICC profile to another.

    Args:
        image: Image to convert
        target_profile: Target ICC profile
        source_profile: Source profile (uses embedded if None)
        intent: Rendering intent

    Returns:
        Converted Image
    """
    # pylint: disable=import-outside-toplevel
    from PIL import ImageCms

    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.color.profiles.icc_profile import ICCProfile

    # pylint: disable=import-outside-toplevel
    from dreamstack.raster.core.pixel import BitDepth, PixelData

    # Get source profile
    if source_profile is None:
        if "icc_profile" in image.metadata.custom:
            source_data = image.metadata.custom["icc_profile"]
            source_profile = ICCProfile(data=source_data)
        else:
            source_profile = ICCProfile.srgb()

    # Convert via PIL
    pil_image = image.to_pil()

    src_cms = source_profile.to_pil_profile()
    dst_cms = target_profile.to_pil_profile()

    # Map intent
    intent_map = {
        RenderingIntent.PERCEPTUAL: ImageCms.Intent.PERCEPTUAL,
        RenderingIntent.RELATIVE_COLORIMETRIC: ImageCms.Intent.RELATIVE_COLORIMETRIC,
        RenderingIntent.SATURATION: ImageCms.Intent.SATURATION,
        RenderingIntent.ABSOLUTE_COLORIMETRIC: ImageCms.Intent.ABSOLUTE_COLORIMETRIC,
    }

    converted = ImageCms.profileToProfile(
        pil_image,
        src_cms,
        dst_cms,
        renderingIntent=intent_map.get(intent, ImageCms.Intent.PERCEPTUAL),
    )

    # Create new image
    result = image.copy()
    array = np.array(converted)

    if array.ndim == 2:
        array = array[:, :, np.newaxis]

    result._pixel_data = PixelData(  # pylint: disable=protected-access
        data=array, pixel_format=result.pixel_format, bit_depth=BitDepth.UINT8
    )

    # Update profile
    result.metadata.custom["icc_profile"] = target_profile.data
    result.metadata.color_profile = target_profile.name

    return result
