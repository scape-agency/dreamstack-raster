# -*- coding: utf-8 -*-

"""Linearize / encode pixel data via its working-space transfer function."""

from __future__ import annotations

import numpy as np

from dreamstack.raster.color.pipeline._internal import RGB_LIKE_FORMATS
from dreamstack.raster.core.pixel.alpha_state import AlphaState
from dreamstack.raster.core.pixel.bit_depth import BitDepth
from dreamstack.raster.core.pixel.gamma_state import GammaState
from dreamstack.raster.core.pixel.pixel_data import PixelData


def _apply_transfer(pixel_data: PixelData, *, encode: bool) -> PixelData:
    """Apply the working-space EOTF (encode=False) or OETF (encode=True).

    Operates only on the RGB color channels; alpha (if present) is passed
    through untouched. Returns a float32 normalized ``PixelData``.
    """
    normalized = pixel_data.to_normalized()
    data = normalized.data.astype(np.float32, copy=True)

    if pixel_data.pixel_format in RGB_LIKE_FORMATS:
        rgb = data[..., :3]
        space = pixel_data.working_space
        # Premultiplied data must be unassociated before applying a non-linear
        # transfer function, otherwise blacks bleed; the caller is expected
        # to pass straight alpha (we assert in ensure()).
        if encode:
            data[..., :3] = space.encode(rgb).astype(np.float32, copy=False)
        else:
            data[..., :3] = space.linearize(rgb).astype(np.float32, copy=False)

    return PixelData(
        data=data,
        pixel_format=pixel_data.pixel_format,
        bit_depth=BitDepth.FLOAT32,
        working_space=pixel_data.working_space,
        gamma_state=GammaState.ENCODED if encode else GammaState.LINEAR,
        alpha_state=pixel_data.alpha_state,
    )


def to_linear(pixel_data: PixelData) -> PixelData:
    """Return a copy in linear-light state.

    No-op (returns a copy) for non-RGB pixel formats and for already-linear
    data. Premultiplied data is unassociated first.
    """
    if pixel_data.gamma_state is GammaState.LINEAR:
        return pixel_data.copy()
    if pixel_data.alpha_state is AlphaState.PREMULTIPLIED:
        # Local import to avoid cycles within the pipeline package.
        from dreamstack.raster.color.pipeline.to_alpha import to_straight

        pixel_data = to_straight(pixel_data)
    return _apply_transfer(pixel_data, encode=False)


def to_encoded(pixel_data: PixelData) -> PixelData:
    """Return a copy display-encoded via the working-space transfer fn.

    Premultiplied data is unassociated first.
    """
    if pixel_data.gamma_state is GammaState.ENCODED:
        return pixel_data.copy()
    if pixel_data.alpha_state is AlphaState.PREMULTIPLIED:
        from dreamstack.raster.color.pipeline.to_alpha import to_straight

        pixel_data = to_straight(pixel_data)
    return _apply_transfer(pixel_data, encode=True)
