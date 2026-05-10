# -*- coding: utf-8 -*-
# pylint: disable=missing-function-docstring

"""Tests for ColorSpace transfer functions and gamma round-trips."""

from __future__ import annotations

import numpy as np
import pytest

from dreamstack.raster.color.spaces import (
    ACEScg,
    AdobeRGB,
    DisplayP3,
    ProPhotoRGB,
    Rec709,
    Rec2020,
    sRGB,
)

WORKING_SPACES = [
    sRGB,
    AdobeRGB,
    ProPhotoRGB,
    DisplayP3,
    Rec709,
    Rec2020,
    ACEScg,
]


@pytest.mark.parametrize("space", WORKING_SPACES, ids=lambda s: s.name)
def test_linearize_encode_round_trip(space, srgb_ramp_f32):
    encoded = srgb_ramp_f32.astype(np.float64)
    linear = space.linearize(encoded)
    re_encoded = space.encode(linear)
    np.testing.assert_allclose(re_encoded, encoded, atol=1e-6)


@pytest.mark.parametrize("space", WORKING_SPACES, ids=lambda s: s.name)
def test_rgb_to_xyz_matrix_well_conditioned(space):
    m = space.rgb_to_xyz_matrix
    inv = space.xyz_to_rgb_matrix
    np.testing.assert_allclose(m @ inv, np.eye(3), atol=1e-10)


def test_srgb_linearize_known_values():
    # sRGB EOTF reference points.
    encoded = np.array([0.0, 0.04045, 0.5, 1.0])
    linear = sRGB.linearize(encoded)
    expected = np.array(
        [
            0.0,
            0.04045 / 12.92,
            ((0.5 + 0.055) / 1.055) ** 2.4,
            1.0,
        ]
    )
    np.testing.assert_allclose(linear, expected, atol=1e-6)
