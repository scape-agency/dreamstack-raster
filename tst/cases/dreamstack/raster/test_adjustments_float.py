# -*- coding: utf-8 -*-
# pylint: disable=missing-function-docstring

"""Float-pipeline adjustment contract tests.

These pin behaviour after Phase 4: ``apply_adjustment`` must operate
on float32 ``[0, 1]`` data without round-tripping through uint8, and
leaf adjustments must respect ``BitDepth.FLOAT32`` semantics.
"""

from __future__ import annotations

import numpy as np
import pytest

from dreamstack.raster.adjustments import (
    ADJUSTMENTS,
    apply_adjustment,
    brightness,
)
from dreamstack.raster.core.image import Image, ImageMetadata
from dreamstack.raster.core.pixel import BitDepth, PixelData, PixelFormat


def _make_image(arr: np.ndarray) -> Image:
    return Image(
        PixelData(
            data=arr,
            pixel_format=PixelFormat.RGBA,
            bit_depth=BitDepth.FLOAT32,
        ),
        ImageMetadata(),
    )


def test_apply_adjustment_preserves_float32_precision():
    """Round-trip should not introduce uint8 quantization."""
    rng = np.random.default_rng(seed=42)
    data = rng.uniform(0.1, 0.9, size=(8, 8, 4)).astype(np.float32)

    # Brightness +0 must be a no-op (within float32 epsilon).
    result = apply_adjustment(data, "brightness", {"amount": 0})

    assert result.dtype == np.float32
    np.testing.assert_allclose(result, data, atol=1e-6)


def test_apply_adjustment_unknown_returns_input():
    data = np.full((4, 4, 4), 0.5, dtype=np.float32)
    out = apply_adjustment(data, "does_not_exist", {})
    assert out is data


def test_brightness_float32_uses_unit_max():
    """A float image should treat 1.0 (not 65535) as full brightness."""
    data = np.full((2, 2, 4), 0.5, dtype=np.float32)
    image = _make_image(data)
    result = brightness(image, amount=50)  # +50% of max
    # max_value for FLOAT32 is 1.0, so offset = 0.5 → result = 1.0
    np.testing.assert_allclose(result.data, 1.0, atol=1e-6)


def test_brightness_uint8_unchanged_behaviour():
    """uint8 path must keep working."""
    data = np.full((2, 2, 4), 128, dtype=np.uint8)
    image = Image(
        PixelData(
            data=data,
            pixel_format=PixelFormat.RGBA,
            bit_depth=BitDepth.UINT8,
        ),
        ImageMetadata(),
    )
    result = brightness(image, amount=50)
    # offset = 50 * 255 / 100 = 127.5 → 255 after clip + cast
    assert result.data.dtype == np.uint8
    assert int(result.data[0, 0, 0]) == 255


@pytest.mark.parametrize("name", sorted(ADJUSTMENTS.keys()))
def test_apply_adjustment_smoke_all_names(name):
    """Each registered adjustment should run on float32 input.

    Uses default parameters (empty kwargs) where possible. Adjustments
    requiring mandatory args are skipped — they are exercised in their
    own dedicated tests.
    """
    data = np.linspace(0, 1, 4 * 4 * 4, dtype=np.float32)
    data = data.reshape(4, 4, 4)  # pylint: disable=too-many-function-args
    try:
        out = apply_adjustment(data, name, {})
    except TypeError:
        pytest.skip(f"{name} requires explicit parameters")
        return
    assert out.dtype == np.float32
    assert out.shape == data.shape
