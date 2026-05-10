# -*- coding: utf-8 -*-
# pylint: disable=missing-function-docstring

"""Focused tests for local single-color and bridge helpers."""

from __future__ import annotations

import numpy as np

from dreamstack.raster.color import (
    Color,
    HSLColorModel,
    HSVColorModel,
    RGBColorModel,
    array_to_rgb,
    convert_color_model,
    create_gradient,
    rgb_to_array,
)


def test_rgb_color_model_hex_and_array_helpers():
    model = RGBColorModel.from_hex("#336699cc")

    np.testing.assert_allclose(
        model.to_array(normalized=True, include_alpha=True),
        [0.2, 0.4, 0.6, 0.8],
        atol=1e-7,
    )
    assert model.to_hsv()[0] == 210.0


def test_bridge_rgb_round_trip_preserves_alpha():
    array = np.array([0.25, 0.5, 0.75, 0.4], dtype=np.float64)

    model = array_to_rgb(array)
    out = rgb_to_array(model, include_alpha=True)

    np.testing.assert_allclose(out, array, atol=1 / 255)


def test_convert_color_model_between_local_models():
    hsv = HSVColorModel(0.0, 100.0, 100.0)
    rgb = convert_color_model(hsv, "rgb")
    hsl = convert_color_model(rgb, "hsl")

    assert rgb == RGBColorModel(255, 0, 0, 1.0)
    assert isinstance(hsl, HSLColorModel)
    np.testing.assert_allclose(
        [hsl.h, hsl.s, hsl.l], [0.0, 100.0, 50.0], atol=1e-7
    )


def test_color_round_trip_hsv_hsl_helpers():
    color = Color.from_hex("#336699")

    hsv_color = Color.from_hsv(*color.to_hsv())
    hsl_color = Color.from_hsl(*color.to_hsl())

    assert hsv_color.to_hex() == color.to_hex()
    assert hsl_color.to_hex() == color.to_hex()


def test_color_manipulation_methods_are_local_and_stable():
    color = Color(64, 128, 192)

    assert color.complement().to_hex() == "#c08040"
    assert color.invert().to_hex() == "#bf7f3f"
    assert (
        color.grayscale().to_rgb()[0]
        == color.grayscale().to_rgb()[1]
        == color.grayscale().to_rgb()[2]
    )
    assert color.lighten(0.2).luminance() > color.luminance()
    assert color.darken(0.2).luminance() < color.luminance()
    assert color.saturate(0.2).to_hsl()[1] > color.to_hsl()[1]
    assert color.desaturate(0.2).to_hsl()[1] < color.to_hsl()[1]


def test_create_gradient_uses_local_color_methods():
    gradient = create_gradient(
        Color.from_hex("#ff0000"),
        Color.from_hex("#00ff00"),
        steps=3,
        color_space="hsv",
    )

    assert gradient.to_hex_list() == ["#ff0000", "#ffff00", "#00ff00"]
