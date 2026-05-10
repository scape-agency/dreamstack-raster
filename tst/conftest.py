# -*- coding: utf-8 -*-
"""Pytest configuration and shared fixtures for dreamstack-raster tests."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """Deterministic random generator for reproducible tests."""
    return np.random.default_rng(seed=20260510)


@pytest.fixture
def srgb_ramp_f32() -> np.ndarray:
    """A 1x256x3 sRGB-encoded float32 ramp in [0,1] (R=G=B = i/255)."""
    v = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    return np.stack([v, v, v], axis=-1)[np.newaxis, :, :]


@pytest.fixture
def srgb_swatches_f32() -> np.ndarray:
    """Canonical sRGB swatches (encoded, float32 in [0,1]).

    Order: black, white, red, green, blue, mid-gray, cyan, magenta, yellow.
    Shape: (1, 9, 3).
    """
    swatches = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.5],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    return swatches[np.newaxis, :, :]
