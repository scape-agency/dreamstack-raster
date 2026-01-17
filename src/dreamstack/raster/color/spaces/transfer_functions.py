# -*- coding: utf-8 -*-

"""Transfer functions for HDR and gamma encoding."""

from __future__ import annotations

import numpy as np


def _pq_eotf(encoded: np.ndarray) -> np.ndarray:
    """PQ (Perceptual Quantizer) EOTF for HDR."""
    m1 = 0.1593017578125
    m2 = 78.84375
    c1 = 0.8359375
    c2 = 18.8515625
    c3 = 18.6875

    encoded = np.maximum(encoded, 0)

    numerator = np.maximum(np.power(encoded, 1 / m2) - c1, 0)
    denominator = c2 - c3 * np.power(encoded, 1 / m2)

    return 10000 * np.power(numerator / denominator, 1 / m1)


def _pq_oetf(linear: np.ndarray) -> np.ndarray:
    """PQ (Perceptual Quantizer) OETF for HDR."""
    m1 = 0.1593017578125
    m2 = 78.84375
    c1 = 0.8359375
    c2 = 18.8515625
    c3 = 18.6875

    linear = np.maximum(linear, 0) / 10000

    numerator = c1 + c2 * np.power(linear, m1)
    denominator = 1 + c3 * np.power(linear, m1)

    return np.power(numerator / denominator, m2)


def _hlg_eotf(encoded: np.ndarray) -> np.ndarray:
    """Hybrid Log-Gamma EOTF."""
    a = 0.17883277
    b = 0.28466892
    c = 0.55991073

    return np.where(
        encoded <= 0.5,
        np.power(encoded, 2) / 3,
        (np.exp((encoded - c) / a) + b) / 12,
    )


def _hlg_oetf(linear: np.ndarray) -> np.ndarray:
    """Hybrid Log-Gamma OETF."""
    a = 0.17883277
    b = 0.28466892
    c = 0.55991073

    return np.where(
        linear <= 1 / 12, np.sqrt(3 * linear), a * np.log(12 * linear - b) + c
    )
