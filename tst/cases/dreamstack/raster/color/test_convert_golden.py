# -*- coding: utf-8 -*-
# pylint: disable=missing-function-docstring

"""Golden-value tests for color conversions vs colour-science."""

from __future__ import annotations

import colour
import numpy as np
import pytest

from dreamstack.raster.color.convert import (
    hsl_to_rgb,
    hsv_to_rgb,
    lab_to_rgb,
    rgb_to_hsl,
    rgb_to_hsv,
    rgb_to_lab,
    rgb_to_xyz,
    xyz_to_rgb,
)
from dreamstack.raster.color.spaces import AdobeRGB

# Tolerance for golden comparisons against colour-science.
# colour-science publishes the IEC 61966-2-1 sRGB matrix to 4 decimal places;
# our internal matrix uses higher-precision Bruce Lindbloom values, so we
# allow ~5e-4 to absorb that published-matrix rounding while still catching
# real algorithmic regressions.
ATOL = 5e-4


# -----------------------------------------------------------------------------
# RGB <-> XYZ (D65, sRGB primaries)
# -----------------------------------------------------------------------------


def test_rgb_to_xyz_matches_colour_science(srgb_swatches_f32):
    rgb = srgb_swatches_f32[0].astype(np.float64)

    expected = colour.sRGB_to_XYZ(rgb)
    actual = rgb_to_xyz(rgb, illuminant="D65")

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=ATOL)


def test_xyz_to_rgb_round_trip(srgb_swatches_f32):
    rgb = srgb_swatches_f32[0].astype(np.float64)
    round_trip = xyz_to_rgb(rgb_to_xyz(rgb, "D65"), "D65")
    np.testing.assert_allclose(round_trip, rgb, atol=1e-3)


def test_rgb_to_xyz_respects_working_space():
    rgb = np.array([0.25, 0.5, 0.75], dtype=np.float64)

    expected = np.einsum(
        "...j,ij->...i",
        AdobeRGB.linearize(rgb),
        AdobeRGB.rgb_to_xyz_matrix,
    )
    actual = rgb_to_xyz(rgb, illuminant="D65", working_space=AdobeRGB)
    srgb_actual = rgb_to_xyz(rgb, illuminant="D65")

    np.testing.assert_allclose(actual, expected, atol=1e-7)
    assert not np.allclose(actual, srgb_actual, atol=1e-4)


def test_xyz_to_rgb_respects_working_space():
    rgb = np.array([0.25, 0.5, 0.75], dtype=np.float64)
    xyz = rgb_to_xyz(rgb, illuminant="D65", working_space=AdobeRGB)

    round_trip = xyz_to_rgb(xyz, illuminant="D65", working_space=AdobeRGB)

    np.testing.assert_allclose(round_trip, rgb, atol=1e-7)


# -----------------------------------------------------------------------------
# RGB <-> Lab
# -----------------------------------------------------------------------------


def test_rgb_to_lab_matches_colour_science(srgb_swatches_f32):
    rgb = srgb_swatches_f32[0].astype(np.float64)

    xyz = colour.sRGB_to_XYZ(rgb)
    expected = colour.XYZ_to_Lab(xyz)
    actual = rgb_to_lab(rgb, illuminant="D65")

    assert actual.shape == expected.shape
    # Lab numbers can range up to ~100; absolute tolerance 0.05 ~= ΔE 0.05.
    np.testing.assert_allclose(actual, expected, atol=0.05)


def test_lab_to_rgb_round_trip(srgb_swatches_f32):
    rgb = srgb_swatches_f32[0].astype(np.float64)
    lab = rgb_to_lab(rgb, "D65")
    rgb_back = lab_to_rgb(lab, "D65")
    np.testing.assert_allclose(rgb_back, rgb, atol=1e-3)


# -----------------------------------------------------------------------------
# RGB <-> HSV
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rgb,expected_h,expected_s,expected_v",
    [
        ((1.0, 0.0, 0.0), 0.0, 1.0, 1.0),  # red
        ((0.0, 1.0, 0.0), 120.0, 1.0, 1.0),  # green
        ((0.0, 0.0, 1.0), 240.0, 1.0, 1.0),  # blue
        ((1.0, 1.0, 0.0), 60.0, 1.0, 1.0),  # yellow
        ((0.0, 1.0, 1.0), 180.0, 1.0, 1.0),  # cyan
        ((1.0, 0.0, 1.0), 300.0, 1.0, 1.0),  # magenta
        ((0.5, 0.5, 0.5), 0.0, 0.0, 0.5),  # gray (h undefined; impl chooses 0)
        ((0.0, 0.0, 0.0), 0.0, 0.0, 0.0),  # black
    ],
)
def test_rgb_to_hsv_known_values(rgb, expected_h, expected_s, expected_v):
    out = rgb_to_hsv(np.array(rgb, dtype=np.float64))
    # Hue may be in [0,1] or [0,360]; detect by max value.
    h = out[0]
    if h <= 1.0 and expected_h > 1.0:
        h = h * 360.0
    np.testing.assert_allclose(
        [h, out[1], out[2]], [expected_h, expected_s, expected_v], atol=1e-4
    )


def test_rgb_hsv_round_trip(srgb_swatches_f32):
    rgb = srgb_swatches_f32[0].astype(np.float64)
    np.testing.assert_allclose(hsv_to_rgb(rgb_to_hsv(rgb)), rgb, atol=1e-5)


# -----------------------------------------------------------------------------
# RGB <-> HSL
# -----------------------------------------------------------------------------


def test_rgb_hsl_round_trip(srgb_swatches_f32):
    rgb = srgb_swatches_f32[0].astype(np.float64)
    np.testing.assert_allclose(hsl_to_rgb(rgb_to_hsl(rgb)), rgb, atol=1e-5)
