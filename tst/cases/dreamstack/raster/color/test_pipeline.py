# -*- coding: utf-8 -*-
# pylint: disable=missing-function-docstring

"""Tests for the typed color pipeline (ensure / to_linear / to_encoded /
to_premultiplied / to_straight / to_working_space)."""

from __future__ import annotations

import numpy as np
import pytest

from dreamstack.raster.color.pipeline import (
    ensure,
    to_encoded,
    to_linear,
    to_premultiplied,
    to_straight,
    to_working_space,
)
from dreamstack.raster.color.spaces import AdobeRGB, DisplayP3, sRGB
from dreamstack.raster.core.pixel import (
    AlphaState,
    BitDepth,
    GammaState,
    PixelData,
    PixelFormat,
)


def _rgba_f32(swatches: np.ndarray, alpha: float = 1.0) -> PixelData:
    rgb = swatches[0]
    a = np.full((rgb.shape[0], 1), alpha, dtype=np.float32)
    rgba = np.concatenate([rgb.astype(np.float32), a], axis=-1)[
        np.newaxis, :, :
    ]
    return PixelData(
        data=rgba,
        pixel_format=PixelFormat.RGBA,
        bit_depth=BitDepth.FLOAT32,
        working_space=sRGB,
        gamma_state=GammaState.ENCODED,
        alpha_state=AlphaState.STRAIGHT,
    )


# -----------------------------------------------------------------------------
# Gamma round-trips
# -----------------------------------------------------------------------------


def test_to_linear_then_to_encoded_round_trip(srgb_swatches_f32):
    pd = _rgba_f32(srgb_swatches_f32)
    back = to_encoded(to_linear(pd))
    assert back.gamma_state is GammaState.ENCODED
    np.testing.assert_allclose(back.data, pd.data, atol=1e-6)


def test_to_linear_changes_state_and_values(srgb_swatches_f32):
    pd = _rgba_f32(srgb_swatches_f32)
    lin = to_linear(pd)
    assert lin.gamma_state is GammaState.LINEAR
    # Mid-gray sRGB ~= 0.5 encoded -> ~0.214 linear.
    mid_lin = lin.data[0, 5, 0]
    assert 0.20 < mid_lin < 0.23


def test_to_linear_is_no_op_when_already_linear(srgb_swatches_f32):
    pd = _rgba_f32(srgb_swatches_f32)
    pd.gamma_state = GammaState.LINEAR
    out = to_linear(pd)
    assert out is not pd
    np.testing.assert_array_equal(out.data, pd.data)


# -----------------------------------------------------------------------------
# Alpha round-trips
# -----------------------------------------------------------------------------


def test_to_premultiplied_then_to_straight_round_trip(srgb_swatches_f32):
    pd = _rgba_f32(srgb_swatches_f32, alpha=0.5)
    back = to_straight(to_premultiplied(pd))
    assert back.alpha_state is AlphaState.STRAIGHT
    np.testing.assert_allclose(back.data, pd.data, atol=1e-6)


def test_to_straight_collapses_zero_alpha(srgb_swatches_f32):
    pd = _rgba_f32(srgb_swatches_f32, alpha=0.0)
    pd.alpha_state = AlphaState.PREMULTIPLIED
    out = to_straight(pd)
    np.testing.assert_array_equal(out.data[..., :3], 0.0)


# -----------------------------------------------------------------------------
# Working-space conversion
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("intermediate", [AdobeRGB, DisplayP3])
def test_to_working_space_round_trip_through_intermediate(
    srgb_swatches_f32, intermediate
):
    pd = _rgba_f32(srgb_swatches_f32)
    via = to_working_space(pd, intermediate)
    back = to_working_space(via, sRGB)
    # Round-trip should be exact within float32; we still need to encode
    # back to compare against the original (input was ENCODED).
    back_encoded = to_encoded(back)
    np.testing.assert_allclose(back_encoded.data, pd.data, atol=1e-5)


def test_to_working_space_rejects_non_rgb_format(srgb_swatches_f32):
    pd = _rgba_f32(srgb_swatches_f32)
    pd.pixel_format = PixelFormat.LAB
    # Channel-count check would normally reject this, but we bypass via
    # direct attribute mutation on a fixture; the pipeline itself should
    # also raise.
    with pytest.raises(ValueError):
        to_working_space(pd, AdobeRGB)


# -----------------------------------------------------------------------------
# ensure() composition
# -----------------------------------------------------------------------------


def test_ensure_full_coercion(srgb_swatches_f32):
    pd = _rgba_f32(srgb_swatches_f32)
    out = ensure(
        pd,
        space=AdobeRGB,
        gamma=GammaState.LINEAR,
        alpha=AlphaState.PREMULTIPLIED,
        dtype=BitDepth.FLOAT32,
    )
    assert out.working_space is AdobeRGB
    assert out.gamma_state is GammaState.LINEAR
    assert out.alpha_state is AlphaState.PREMULTIPLIED
    assert out.bit_depth is BitDepth.FLOAT32


def test_ensure_no_change_returns_copy(srgb_swatches_f32):
    pd = _rgba_f32(srgb_swatches_f32)
    out = ensure(pd)
    assert out is not pd
    np.testing.assert_array_equal(out.data, pd.data)
    out.data[...] = 0.0  # mutate copy, original must be untouched
    assert pd.data.max() > 0
