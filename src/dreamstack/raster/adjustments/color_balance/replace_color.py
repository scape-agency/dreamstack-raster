# -*- coding: utf-8 -*-

"""Replace color function."""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np

if TYPE_CHECKING:
    from dreamstack.raster.core.image import Image

from dreamstack.raster.adjustments.color_balance._color_utils import (
    _hsl_to_rgb,
    _rgb_to_hsl,
)


def replace_color(
    image: Image,
    source_color: Tuple[int, int, int],
    target_color: Tuple[int, int, int],
    fuzziness: float = 40,
    hue_shift: float = 0,
    saturation_shift: float = 0,
    lightness_shift: float = 0,
) -> Image:
    """
    Replace a specific color with another.

    Args:
        image: Input image
        source_color: Color to replace (R, G, B)
        target_color: Replacement color (R, G, B) - ignored if shifts are used
        fuzziness: Color matching tolerance (0-200)
        hue_shift: Hue adjustment instead of replacement
        saturation_shift: Saturation adjustment
        lightness_shift: Lightness adjustment

    Returns:
        Adjusted image
    """
    from dreamstack.raster.core.pixel import PixelData

    if image.channels < 3:
        return image.copy()

    data = image.data.astype(np.float32)
    max_val = 255 if image.bit_depth.name == "UINT8" else 65535

    # Normalize
    normalized = data[:, :, :3] / max_val
    source_norm = np.array(source_color) / 255

    # Calculate color distance
    distance = np.sqrt(np.sum((normalized - source_norm) ** 2, axis=2))

    # Create selection mask
    tolerance = fuzziness / 100
    mask = np.clip(1 - distance / tolerance, 0, 1)

    if hue_shift != 0 or saturation_shift != 0 or lightness_shift != 0:
        # Apply HSL shifts
        hsl = _rgb_to_hsl(normalized)

        hsl[:, :, 0] = (hsl[:, :, 0] + hue_shift / 360.0 * mask) % 1.0

        if saturation_shift > 0:
            hsl[:, :, 1] = (
                hsl[:, :, 1]
                + (1 - hsl[:, :, 1]) * saturation_shift / 100 * mask
            )
        else:
            hsl[:, :, 1] = hsl[:, :, 1] * (1 + saturation_shift / 100 * mask)

        if lightness_shift > 0:
            hsl[:, :, 2] = (
                hsl[:, :, 2]
                + (1 - hsl[:, :, 2]) * lightness_shift / 100 * mask
            )
        else:
            hsl[:, :, 2] = hsl[:, :, 2] * (1 + lightness_shift / 100 * mask)

        result_rgb = _hsl_to_rgb(np.clip(hsl, 0, 1))
    else:
        # Direct color replacement
        target_norm = np.array(target_color) / 255
        result_rgb = (
            normalized * (1 - mask[:, :, np.newaxis])
            + target_norm * mask[:, :, np.newaxis]
        )

    result = data.copy()
    result[:, :, :3] = result_rgb * max_val
    result = np.clip(result, 0, max_val)

    result_image = image.copy()
    result_image._pixel_data = PixelData(
        data=result.astype(image.data.dtype),
        pixel_format=image.pixel_format,
        bit_depth=image.bit_depth,
    )

    return result_image
