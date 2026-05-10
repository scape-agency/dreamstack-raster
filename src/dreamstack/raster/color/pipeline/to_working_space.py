# -*- coding: utf-8 -*-

"""Move pixel data into a different working color space."""

from __future__ import annotations

import numpy as np

from dreamstack.raster.color.pipeline._internal import RGB_LIKE_FORMATS
from dreamstack.raster.color.pipeline.to_alpha import to_straight
from dreamstack.raster.color.pipeline.to_gamma import to_linear
from dreamstack.raster.color.spaces.color_space import ColorSpace
from dreamstack.raster.core.pixel.alpha_state import AlphaState
from dreamstack.raster.core.pixel.bit_depth import BitDepth
from dreamstack.raster.core.pixel.gamma_state import GammaState
from dreamstack.raster.core.pixel.pixel_data import PixelData


def to_working_space(pixel_data: PixelData, target: ColorSpace) -> PixelData:
    """Convert RGB-like ``PixelData`` to a different working ``ColorSpace``.

    Conversion is performed in linear-light, straight-alpha float32 via
    a linear-XYZ pivot. The returned ``PixelData`` is in linear gamma
    state in the *target* working space; callers that want the target
    space's display-encoded form should follow up with :func:`to_encoded`.

    Non-RGB pixel formats (LAB / HSV / CMYK / GRAY*) raise ``ValueError``;
    convert them via the format-conversion routines first.
    """
    if pixel_data.pixel_format not in RGB_LIKE_FORMATS:
        raise ValueError(
            f"to_working_space requires an RGB-like pixel format, got "
            f"{pixel_data.pixel_format.name}"
        )

    # Same-space short-circuit (object identity OR equal name+gamma).
    src = pixel_data.working_space
    if src is target or (
        src.name == target.name and src.gamma_type == target.gamma_type
    ):
        return pixel_data.copy()

    # Move to a known canonical state: float32 linear straight.
    work = to_straight(to_linear(pixel_data))
    rgb = work.data[..., :3]

    m = target.xyz_to_rgb_matrix @ src.rgb_to_xyz_matrix
    converted = np.einsum("...j,ij->...i", rgb, m).astype(
        np.float32, copy=False
    )

    if work.has_alpha:
        out = np.concatenate([converted, work.data[..., 3:4]], axis=-1)
    else:
        out = converted

    return PixelData(
        data=out,
        pixel_format=work.pixel_format,
        bit_depth=BitDepth.FLOAT32,
        working_space=target,
        gamma_state=GammaState.LINEAR,
        alpha_state=AlphaState.STRAIGHT,
    )
