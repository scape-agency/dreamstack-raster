# -*- coding: utf-8 -*-
# pylint: disable=missing-function-docstring,too-many-function-args

"""Float-pipeline compositing contract tests.

Pin Phase 5: ``blend`` operates internally on float32 and preserves
input dtype on output; ``blend_float`` is a pure float32 RGB kernel;
algebraic identities hold to float precision (no uint8 quantization).
"""

from __future__ import annotations

import numpy as np
import pytest

from dreamstack.raster.compositing.blend import BlendMode, blend, blend_float

# ---------------------------------------------------------------------------
# Identity / algebraic properties
# ---------------------------------------------------------------------------


def test_blend_normal_returns_overlay():
    base = np.full((4, 4, 3), 0.25, dtype=np.float32)
    overlay = np.full((4, 4, 3), 0.75, dtype=np.float32)
    out = blend_float(base, overlay, BlendMode.NORMAL)
    np.testing.assert_allclose(out, overlay, atol=1e-7)


def test_blend_multiply_with_white_is_identity():
    base = (
        np.linspace(0, 1, 16, dtype=np.float32)
        .reshape(4, 4, 1)
        .repeat(3, axis=2)
    )
    white = np.ones_like(base)
    out = blend_float(base, white, BlendMode.MULTIPLY)
    np.testing.assert_allclose(out, base, atol=1e-7)


def test_blend_multiply_with_black_is_zero():
    base = (
        np.linspace(0, 1, 16, dtype=np.float32)
        .reshape(4, 4, 1)
        .repeat(3, axis=2)
    )
    black = np.zeros_like(base)
    out = blend_float(base, black, BlendMode.MULTIPLY)
    np.testing.assert_allclose(out, 0.0, atol=1e-7)


def test_blend_screen_with_black_is_identity():
    base = (
        np.linspace(0, 1, 16, dtype=np.float32)
        .reshape(4, 4, 1)
        .repeat(3, axis=2)
    )
    black = np.zeros_like(base)
    out = blend_float(base, black, BlendMode.SCREEN)
    np.testing.assert_allclose(out, base, atol=1e-7)


def test_blend_screen_with_white_is_white():
    base = (
        np.linspace(0, 1, 16, dtype=np.float32)
        .reshape(4, 4, 1)
        .repeat(3, axis=2)
    )
    white = np.ones_like(base)
    out = blend_float(base, white, BlendMode.SCREEN)
    np.testing.assert_allclose(out, 1.0, atol=1e-7)


def test_blend_difference_with_self_is_zero():
    rng = np.random.default_rng(seed=0)
    img = rng.uniform(0, 1, size=(8, 8, 3)).astype(np.float32)
    out = blend_float(img, img, BlendMode.DIFFERENCE)
    np.testing.assert_allclose(out, 0.0, atol=1e-7)


def test_blend_opacity_zero_is_passthrough():
    base = np.full((4, 4, 3), 0.3, dtype=np.float32)
    overlay = np.full((4, 4, 3), 0.9, dtype=np.float32)
    out = blend_float(base, overlay, BlendMode.MULTIPLY, opacity=0.0)
    np.testing.assert_allclose(out, base, atol=1e-7)


# ---------------------------------------------------------------------------
# Dtype preservation
# ---------------------------------------------------------------------------


def test_blend_preserves_uint8_dtype():
    base = np.full((4, 4, 3), 128, dtype=np.uint8)
    overlay = np.full((4, 4, 3), 64, dtype=np.uint8)
    out = blend(base, overlay, BlendMode.NORMAL)
    assert out.dtype == np.uint8
    assert int(out[0, 0, 0]) == 64


def test_blend_preserves_float32_dtype_no_quantization():
    """Float blends must NOT round-trip through uint8.

    A subtle multiply that cannot be represented in 8-bit precision
    proves the new pipeline keeps full precision.
    """
    base = np.full((2, 2, 3), 0.501960, dtype=np.float32)  # ~ 128/255
    overlay = np.full((2, 2, 3), 0.501960, dtype=np.float32)
    out = blend(base, overlay, BlendMode.MULTIPLY)
    assert out.dtype == np.float32
    expected = 0.501960 * 0.501960
    np.testing.assert_allclose(out, expected, atol=1e-6)


def test_blend_preserves_alpha_channel_uint8():
    base = np.zeros((4, 4, 4), dtype=np.uint8)
    base[..., 3] = 200  # distinctive alpha
    overlay = np.full((4, 4, 4), 255, dtype=np.uint8)
    out = blend(base, overlay, BlendMode.NORMAL)
    assert out.shape == base.shape
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out[..., 3], 200)


def test_blend_preserves_alpha_channel_float32():
    base = np.zeros((4, 4, 4), dtype=np.float32)
    base[..., 3] = 0.4
    overlay = np.ones((4, 4, 4), dtype=np.float32)
    out = blend(base, overlay, BlendMode.NORMAL)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out[..., 3], 0.4, atol=1e-7)


# ---------------------------------------------------------------------------
# Smoke: every mode produces finite output in [0, 1]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", list(BlendMode))
def test_blend_float_all_modes_in_range(mode):
    rng = np.random.default_rng(seed=1)
    base = rng.uniform(0, 1, size=(4, 4, 3)).astype(np.float32)
    overlay = rng.uniform(0, 1, size=(4, 4, 3)).astype(np.float32)
    out = blend_float(base, overlay, mode)
    assert out.dtype == np.float32
    assert out.shape == base.shape
    assert np.isfinite(out).all()
    assert (out >= 0.0).all() and (out <= 1.0).all()
