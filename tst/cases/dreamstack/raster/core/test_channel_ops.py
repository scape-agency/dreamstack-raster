# -*- coding: utf-8 -*-
# pylint: disable=redefined-outer-name,missing-function-docstring

"""Tests for the unified RGBA-only channel operations."""

from __future__ import annotations

import numpy as np
import pytest

from dreamstack.raster.core.channel import (
    channel_to_grayscale_rgb,
    extract_channel,
    extract_rgb_arrays,
    isolate_channel,
    merge_channels,
    split_channels,
    swap_channels,
)


@pytest.fixture
def rgba_u8() -> np.ndarray:
    return np.array(
        [[[10, 20, 30, 40], [50, 60, 70, 80]]],
        dtype=np.uint8,
    )


def test_split_then_merge_round_trip(rgba_u8):
    parts = split_channels(rgba_u8)
    assert len(parts) == 4
    assert all(p.shape == (1, 2) for p in parts)
    rebuilt = merge_channels(parts)
    np.testing.assert_array_equal(rebuilt, rgba_u8)


@pytest.mark.parametrize(
    "name,index,expected",
    [("red", 0, 10), ("g", 1, 20), ("blue", 2, 30), ("a", 3, 40)],
)
def test_extract_channel_by_name_and_index(rgba_u8, name, index, expected):
    np.testing.assert_array_equal(
        extract_channel(rgba_u8, name), extract_channel(rgba_u8, index)
    )
    assert extract_channel(rgba_u8, name)[0, 0] == expected


def test_extract_channel_unknown_name_raises(rgba_u8):
    with pytest.raises(ValueError):
        extract_channel(rgba_u8, "yellow")  # type: ignore[arg-type]


def test_isolate_channel_zeros_others_keeps_alpha(rgba_u8):
    out = isolate_channel(rgba_u8, "red")
    np.testing.assert_array_equal(out[..., 0], rgba_u8[..., 0])
    np.testing.assert_array_equal(out[..., 1], 0)
    np.testing.assert_array_equal(out[..., 2], 0)
    np.testing.assert_array_equal(out[..., 3], rgba_u8[..., 3])


def test_swap_channels(rgba_u8):
    swapped = swap_channels(rgba_u8, "red", "blue")
    np.testing.assert_array_equal(swapped[..., 0], rgba_u8[..., 2])
    np.testing.assert_array_equal(swapped[..., 2], rgba_u8[..., 0])


def test_channel_to_grayscale_rgb_promotes_2d():
    ch = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    out = channel_to_grayscale_rgb(ch)
    assert out.shape == (2, 2, 3)
    for i in range(3):
        np.testing.assert_array_equal(out[..., i], ch)


def test_extract_rgb_arrays_returns_three_colorized(rgba_u8):
    r, g, b = extract_rgb_arrays(rgba_u8)
    assert r.shape == g.shape == b.shape == (1, 2, 3)
    np.testing.assert_array_equal(r[..., 0], rgba_u8[..., 0])
    np.testing.assert_array_equal(r[..., 1], 0)
    np.testing.assert_array_equal(r[..., 2], 0)


def test_split_channels_grayscale_2d():
    g = np.zeros((4, 4), dtype=np.float32)
    parts = split_channels(g)
    assert len(parts) == 1
    assert parts[0].shape == (4, 4)


def test_dtype_is_preserved():
    img = np.zeros((2, 2, 3), dtype=np.uint16)
    assert split_channels(img)[0].dtype == np.uint16
    assert merge_channels(split_channels(img)).dtype == np.uint16
    assert isolate_channel(img, "red").dtype == np.uint16
