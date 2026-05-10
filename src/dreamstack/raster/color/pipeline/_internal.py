# -*- coding: utf-8 -*-

"""Internal helpers shared by the color pipeline transforms."""

from __future__ import annotations

import numpy as np

from dreamstack.raster.core.pixel.pixel_data import PixelData
from dreamstack.raster.core.pixel.pixel_format import PixelFormat

# Pixel formats whose color channels are RGB values that live in a working
# color space and obey a transfer function. Other formats (LAB / HSV / HSL /
# CMYK / GRAY*) carry their own implicit encoding and are not touched by
# gamma / working-space transitions.
RGB_LIKE_FORMATS: frozenset[PixelFormat] = frozenset(
    {PixelFormat.RGB, PixelFormat.RGBA}
)


def split_rgb_alpha(
    pixel_data: PixelData,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return ``(rgb, alpha_or_none)`` views of an RGB-like ``PixelData``.

    ``rgb`` is always a 3-channel float view; ``alpha`` is a single-channel
    float view (kept 2-D, no trailing channel axis) or ``None`` when the
    pixel format has no alpha. Caller is responsible for the dtype contract.
    """
    data = pixel_data.data
    if pixel_data.has_alpha:
        return data[..., :3], data[..., 3]
    return data[..., :3] if data.shape[-1] >= 3 else data, None
