# -*- coding: utf-8 -*-

"""Premultiply / unassociate alpha on RGBA pixel data."""

from __future__ import annotations

import numpy as np

from dreamstack.raster.core.pixel.alpha_state import AlphaState
from dreamstack.raster.core.pixel.bit_depth import BitDepth
from dreamstack.raster.core.pixel.pixel_data import PixelData

_EPS = 1.0 / 65535.0


def to_premultiplied(pixel_data: PixelData) -> PixelData:
    """Return a copy with RGB channels premultiplied by alpha.

    No-op (returns a copy) for formats without an alpha channel and for
    already-premultiplied data. Always returns float32 normalized data.
    """
    if not pixel_data.has_alpha:
        return pixel_data.copy()
    if pixel_data.alpha_state is AlphaState.PREMULTIPLIED:
        return pixel_data.copy()

    normalized = pixel_data.to_normalized()
    data = normalized.data.astype(np.float32, copy=True)
    alpha = data[..., -1:]
    data[..., :-1] *= alpha

    return PixelData(
        data=data,
        pixel_format=pixel_data.pixel_format,
        bit_depth=BitDepth.FLOAT32,
        working_space=pixel_data.working_space,
        gamma_state=pixel_data.gamma_state,
        alpha_state=AlphaState.PREMULTIPLIED,
    )


def to_straight(pixel_data: PixelData) -> PixelData:
    """Return a copy with RGB channels unassociated from alpha.

    No-op (returns a copy) for formats without an alpha channel and for
    already-straight data. Pixels with alpha below ``_EPS`` collapse to 0
    to avoid amplifying noise from a divide-by-tiny-alpha.
    """
    if not pixel_data.has_alpha:
        return pixel_data.copy()
    if pixel_data.alpha_state is AlphaState.STRAIGHT:
        return pixel_data.copy()

    normalized = pixel_data.to_normalized()
    data = normalized.data.astype(np.float32, copy=True)
    alpha = data[..., -1:]
    safe = alpha > _EPS
    data[..., :-1] = np.where(
        safe, data[..., :-1] / np.where(safe, alpha, 1.0), 0.0
    )

    return PixelData(
        data=data,
        pixel_format=pixel_data.pixel_format,
        bit_depth=BitDepth.FLOAT32,
        working_space=pixel_data.working_space,
        gamma_state=pixel_data.gamma_state,
        alpha_state=AlphaState.STRAIGHT,
    )
