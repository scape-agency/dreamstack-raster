# -*- coding: utf-8 -*-

"""Single coercion entry point for the typed color pipeline."""

from __future__ import annotations

import numpy as np

from dreamstack.raster.color.pipeline.to_alpha import (
    to_premultiplied,
    to_straight,
)
from dreamstack.raster.color.pipeline.to_gamma import to_encoded, to_linear
from dreamstack.raster.color.pipeline.to_working_space import to_working_space
from dreamstack.raster.color.spaces.color_space import ColorSpace
from dreamstack.raster.core.pixel.alpha_state import AlphaState
from dreamstack.raster.core.pixel.bit_depth import BitDepth
from dreamstack.raster.core.pixel.gamma_state import GammaState
from dreamstack.raster.core.pixel.pixel_data import PixelData


def ensure(
    pixel_data: PixelData,
    *,
    space: ColorSpace | None = None,
    gamma: GammaState | None = None,
    alpha: AlphaState | None = None,
    dtype: BitDepth | None = None,
) -> PixelData:
    """Coerce ``pixel_data`` into the requested color / encoding state.

    Any argument left ``None`` is preserved from the input. The order of
    operations (working space → gamma → alpha → bit depth) is chosen so
    each step sees the canonical inputs the previous step established.

    Always returns a new ``PixelData`` (never mutates the input). When
    every requested state already matches, the input is copied to keep
    the contract uniform for callers.
    """
    out = pixel_data

    if space is not None:
        out = to_working_space(out, space)

    if gamma is not None and gamma is not out.gamma_state:
        out = to_linear(out) if gamma is GammaState.LINEAR else to_encoded(out)

    if alpha is not None and alpha is not out.alpha_state:
        out = (
            to_premultiplied(out)
            if alpha is AlphaState.PREMULTIPLIED
            else to_straight(out)
        )

    if dtype is not None and dtype is not out.bit_depth:
        out = out.to_bit_depth(dtype)

    if out is pixel_data:
        # Caller asked for what we already have; still hand back a fresh copy
        # so they can mutate without surprising the original owner.
        out = pixel_data.copy()

    # Sanity: if we touched anything at all, our cooperators above already
    # produced a fresh array. Defensive assert to catch future regressions
    # cheaply when running tests.
    assert isinstance(out.data, np.ndarray)
    return out
