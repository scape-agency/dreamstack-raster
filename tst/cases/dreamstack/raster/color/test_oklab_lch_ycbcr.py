# -*- coding: utf-8 -*-
# pylint: disable=missing-function-docstring

"""Tests for OKLab, OKLCh, LCh, YCbCr conversions."""

from __future__ import annotations

import numpy as np
import pytest

from dreamstack.raster.color.convert import (
    lab_to_lch,
    lab_to_rgb,
    lch_to_lab,
    oklab_to_oklch,
    oklab_to_rgb,
    oklch_to_oklab,
    rgb_to_lab,
    rgb_to_oklab,
    rgb_to_ycbcr,
    ycbcr_to_rgb,
)

# -----------------------------------------------------------------------------
# OKLab — Ottosson reference values (https://bottosson.github.io/posts/oklab/)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "linear_rgb,expected_oklab",
    [
        ((1.0, 0.0, 0.0), (0.62795553, 0.22486306, 0.12584630)),
        ((0.0, 1.0, 0.0), (0.86643962, -0.23388755, 0.17949837)),
        ((0.0, 0.0, 1.0), (0.45201370, -0.03245698, -0.31152136)),
        ((1.0, 1.0, 1.0), (1.00000000, 0.00000000, 0.00000000)),
    ],
)
def test_rgb_to_oklab_known_values(linear_rgb, expected_oklab):
    out = rgb_to_oklab(np.array(linear_rgb), linear=True)
    np.testing.assert_allclose(out, expected_oklab, atol=1e-5)


def test_oklab_round_trip(srgb_swatches_f32):
    rgb = srgb_swatches_f32[0].astype(np.float64)
    back = oklab_to_rgb(rgb_to_oklab(rgb))
    np.testing.assert_allclose(back, rgb, atol=1e-5)


def test_oklab_white_is_unit_l():
    out = rgb_to_oklab(np.array([1.0, 1.0, 1.0]), linear=True)
    assert out[0] == pytest.approx(1.0, abs=1e-6)
    np.testing.assert_allclose(out[1:], [0.0, 0.0], atol=1e-6)


# -----------------------------------------------------------------------------
# LCh polar form
# -----------------------------------------------------------------------------


def test_lab_lch_round_trip(srgb_swatches_f32):
    rgb = srgb_swatches_f32[0].astype(np.float64)
    lab = rgb_to_lab(rgb)
    back = lch_to_lab(lab_to_lch(lab))
    np.testing.assert_allclose(back, lab, atol=1e-9)


def test_lch_hue_in_zero_to_360():
    # Pure red sRGB → hue around 39° in CIE LCh(ab).
    lch = lab_to_lch(rgb_to_lab(np.array([[1.0, 0.0, 0.0]])))
    h = lch[0, 2]
    assert 0.0 <= h < 360.0
    assert 35.0 < h < 45.0


def test_oklch_aliases_match_lab_lch():
    sample = np.array([[0.6, 0.1, -0.2]])
    np.testing.assert_array_equal(oklab_to_oklch(sample), lab_to_lch(sample))
    np.testing.assert_array_equal(oklch_to_oklab(sample), lch_to_lab(sample))


def test_lab_round_trip_through_lch_via_rgb(srgb_swatches_f32):
    rgb = srgb_swatches_f32[0].astype(np.float64)
    lab = rgb_to_lab(rgb)
    lch = lab_to_lch(lab)
    rgb_back = lab_to_rgb(lch_to_lab(lch))
    np.testing.assert_allclose(rgb_back, rgb, atol=1e-3)


# -----------------------------------------------------------------------------
# YCbCr
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("standard", ["rec601", "rec709", "rec2020"])
def test_ycbcr_round_trip(srgb_swatches_f32, standard):
    rgb = srgb_swatches_f32[0].astype(np.float64)
    back = ycbcr_to_rgb(rgb_to_ycbcr(rgb, standard), standard)
    np.testing.assert_allclose(back, rgb, atol=1e-12)


def test_ycbcr_white_is_zero_chroma():
    out = rgb_to_ycbcr(np.array([1.0, 1.0, 1.0]))
    assert out[0] == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(out[1:], [0.0, 0.0], atol=1e-12)


def test_ycbcr_alpha_passthrough():
    rgba = np.array([[1.0, 0.5, 0.25, 0.7]])
    out = rgb_to_ycbcr(rgba)
    assert out.shape == (1, 4)
    np.testing.assert_array_equal(out[..., 3], rgba[..., 3])
